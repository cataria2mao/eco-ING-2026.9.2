import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.types import Command, Interrupt

from contextlib import asynccontextmanager

import agent_core
from agent_core import KNOWLEDGE_BASES, init_agent   # 模型 / 技能导入时初始化，图在 startup 构建
from database import Conversation, Message, SessionLocal, User
from auth import (
    authenticate_ws_token,
    create_token,
    get_current_user,
    get_db,
    hash_password,
    verify_password,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 在当前事件循环内构建 AsyncPostgresSaver + 图（幂等）
    await init_agent()
    yield


app = FastAPI(title="生态调查 Agent", lifespan=lifespan)

# values 负责把消息去重后推给前端；updates 负责推送节点级事件（路由/检索/分发）；
# custom 负责推送节点执行过程中的细粒度阶段事件（如检索：混合检索→RRF→精排）
STREAM_MODES = ["values", "updates", "custom"]


# ============================================================
# 序列化工具
# ============================================================
def _jsonable(obj):
    return json.loads(json.dumps(obj, ensure_ascii=False, default=str))


def _serialize_message(msg):
    if isinstance(msg, AIMessage):
        role = "assistant"
    elif isinstance(msg, ToolMessage):
        role = "tool"
    elif isinstance(msg, SystemMessage):
        role = "system"
    else:
        role = "unknown"
    content = msg.content
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False, default=str)
    tool_calls = None
    if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
        tool_calls = _jsonable([{"name": tc["name"], "args": tc["args"]} for tc in msg.tool_calls])
    return {
        "id": getattr(msg, "id", None),
        "role": role,
        "content": content,
        "tool_calls": tool_calls,
    }


def _user_out(u: User) -> dict:
    return {"id": u.id, "username": u.username, "display_name": u.display_name or u.username}


def _conv_out(c: Conversation) -> dict:
    return {
        "id": c.id,
        "thread_id": c.thread_id,
        "title": c.title,
        "updated_at": c.updated_at.isoformat() if c.updated_at else None,
    }


def _msg_out(m: Message) -> dict:
    return {"id": m.message_id, "role": m.role, "content": m.content, "tool_calls": m.tool_calls}


async def _collect_interrupts(config):
    """读取当前挂起的 interrupt（人工审核 / 参数缺失 / 执行出错）"""
    state = await agent_core.parent_graph.aget_state(config)
    out = []
    for task in state.tasks:
        if hasattr(task, "interrupts") and task.interrupts:
            for intr in task.interrupts:
                out.append(intr.value if isinstance(intr, Interrupt) else intr)
    return out


# ============================================================
# 数据库写入辅助
# ============================================================
def _save_message(db: Session, conversation_id: int, message_id: Optional[str], role: str,
                  content: str, tool_calls=None) -> None:
    """按 (conversation_id, message_id) 幂等写入一条消息。"""
    message_id = message_id or uuid.uuid4().hex
    try:
        db.add(Message(
            conversation_id=conversation_id,
            message_id=str(message_id),
            role=role,
            content=content or "",
            tool_calls=tool_calls,
        ))
        db.commit()
    except IntegrityError:
        db.rollback()   # 已存在，忽略


def _touch_conversation(db: Session, conv: Conversation, user_text: Optional[str] = None) -> None:
    """更新会话时间；首次用户发言时用其内容生成标题。"""
    if user_text and (not conv.title or conv.title == "新会话"):
        conv.title = user_text.strip().replace("\n", " ")[:20] or "新会话"
    conv.updated_at = datetime.now(timezone.utc)
    db.commit()


# ============================================================
# 认证接口
# ============================================================
class RegisterIn(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=6, max_length=128)
    display_name: Optional[str] = Field(default=None, max_length=32)


class LoginIn(BaseModel):
    username: str
    password: str


class RenameIn(BaseModel):
    title: str = Field(min_length=1, max_length=80)


def _auth_payload(user: User) -> dict:
    return {"token": create_token(user), "user": _user_out(user)}


@app.post("/api/register")
def register(body: RegisterIn, db: Session = Depends(get_db)):
    username = body.username.strip()
    exists = db.execute(select(User).where(User.username == username)).scalar_one_or_none()
    if exists:
        raise HTTPException(status_code=400, detail="用户名已被占用")
    user = User(
        username=username,
        password_hash=hash_password(body.password),
        display_name=(body.display_name or username).strip(),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return _auth_payload(user)


@app.post("/api/login")
def login(body: LoginIn, db: Session = Depends(get_db)):
    user = db.execute(select(User).where(User.username == body.username.strip())).scalar_one_or_none()
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    return _auth_payload(user)


@app.get("/api/me")
def me(user: User = Depends(get_current_user)):
    return _user_out(user)


# ============================================================
# 会话 / 消息接口
# ============================================================
def _get_owned_conversation(db: Session, user: User, cid: int) -> Conversation:
    conv = db.get(Conversation, cid)
    if conv is None or conv.user_id != user.id:
        raise HTTPException(status_code=404, detail="会话不存在")
    return conv


@app.get("/api/conversations")
def list_conversations(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = db.execute(
        select(Conversation)
        .where(Conversation.user_id == user.id)
        .order_by(Conversation.updated_at.desc())
    ).scalars().all()
    return [_conv_out(c) for c in rows]


@app.post("/api/conversations")
def create_conversation(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    conv = Conversation(
        user_id=user.id,
        thread_id=f"{user.id}-{uuid.uuid4().hex[:16]}",
        title="新会话",
    )
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return _conv_out(conv)


@app.get("/api/conversations/{cid}/messages")
def conversation_messages(cid: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    conv = _get_owned_conversation(db, user, cid)
    rows = db.execute(
        select(Message).where(Message.conversation_id == conv.id).order_by(Message.id.asc())
    ).scalars().all()
    return [_msg_out(m) for m in rows]


@app.patch("/api/conversations/{cid}")
def rename_conversation(cid: int, body: RenameIn, user: User = Depends(get_current_user),
                        db: Session = Depends(get_db)):
    conv = _get_owned_conversation(db, user, cid)
    conv.title = body.title.strip()
    db.commit()
    return _conv_out(conv)


@app.delete("/api/conversations/{cid}")
def delete_conversation(cid: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    conv = _get_owned_conversation(db, user, cid)
    db.execute(Message.__table__.delete().where(Message.conversation_id == conv.id))
    db.delete(conv)
    db.commit()
    return {"ok": True}


# ============================================================
# 节点级事件推送
# ============================================================
async def _emit_node_event(node: str, update, ctx: dict, ws: WebSocket):
    """把关键节点的执行信息（需求判断 / 知识库检索 / 任务分发）推送给前端。"""
    if not isinstance(update, dict):
        return
    try:
        # 1) 需求判断：告诉前端这一轮走的是动物 / 植物子图
        if node == "parent_router":
            route = update.get("route")
            if route in ("animal", "plant"):
                await ws.send_json({
                    "type": "route",
                    "route": route,
                    "kb": update.get("kb") or "none",
                    "reason": update.get("route_reason") or "",
                })

        # 2) 知识库检索：告诉前端查的是哪个库、命中多少条
        elif node == "parent_retrieve":
            kb = ctx.get("kb") or "none"
            cfg = KNOWLEDGE_BASES.get(kb) or {}
            docs = update.get("retrieved_docs") or []
            await ws.send_json({
                "type": "retrieve",
                "kb": kb,
                "label": cfg.get("label", kb),
                "count": len(docs),
            })

        # 3) 任务分发：告诉前端任务已交给哪个子代理
        elif node == "parent_dispatch":
            req_animal = update.get("animal_request") or ""
            req_plant = update.get("plant_request") or ""
            if req_animal:
                domain, agent, label, task = "animal", "animal_base_work_agent", "陆生动物", req_animal
            elif req_plant:
                domain, agent, label, task = "plant", "plant_base_work_agent", "陆生植物", req_plant
            else:
                return
            await ws.send_json({
                "type": "dispatch",
                "domain": domain,
                "agent": agent,
                "label": label,
                "task": task,
            })
    except Exception as e:
        print(f"[WS] 推送节点事件失败({node}): {e}")


async def _stream_and_forward(payload, config, ws, sent_ids, db, conversation_id):
    """把图执行过程按消息 id 去重后逐条发给前端，同时推送节点级事件并落库。"""
    ctx = {"kb": "none", "route": None}

    async for item in agent_core.parent_graph.astream(payload, config=config, stream_mode=STREAM_MODES):
        # 多 stream_mode 时产出 (mode, chunk) 元组；做一次兼容解包
        if isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[0], str):
            mode, chunk = item
        else:
            mode, chunk = "values", item

        if mode == "values":
            if not isinstance(chunk, dict):
                continue
            # 记录上下文，供 updates 事件使用（updates 里拿不到完整状态）
            if "kb" in chunk:
                ctx["kb"] = chunk.get("kb") or "none"
            if "route" in chunk:
                ctx["route"] = chunk.get("route")

            for msg in chunk.get("messages", []) or []:
                if isinstance(msg, HumanMessage):
                    continue  # 用户消息已单独落库并回显
                mid = getattr(msg, "id", None)
                if mid is None or mid in sent_ids:
                    continue
                sent_ids.add(mid)
                serialized = _serialize_message(msg)
                _save_message(db, conversation_id, mid, serialized["role"],
                              serialized["content"], serialized["tool_calls"])
                await ws.send_json({"type": "update", "message": serialized})

        elif mode == "updates":
            for node, update in (chunk or {}).items():
                await _emit_node_event(node, update, ctx, ws)

        elif mode == "custom":
            # 节点内部通过 get_stream_writer() 主动推送的阶段事件
            if not isinstance(chunk, dict):
                continue
            if chunk.get("__retrieve_stage__"):
                await ws.send_json({
                    "type": "retrieve_stage",
                    "kb": chunk.get("kb") or "none",
                    "label": chunk.get("label") or "",
                    "stage": chunk.get("stage") or "",
                    "candidates": chunk.get("candidates"),
                })


# ============================================================
# WebSocket 主循环
# ============================================================
@app.websocket("/ws/{thread_id}")
async def ws_chat(ws: WebSocket, thread_id: str, token: str = Query("")):
    user, db = authenticate_ws_token(token)
    if user is None:
        await ws.close(code=4401)
        db.close()
        return

    conv = db.execute(
        select(Conversation).where(
            Conversation.thread_id == thread_id,
            Conversation.user_id == user.id,
        )
    ).scalar_one_or_none()
    if conv is None:
        await ws.close(code=4404)
        db.close()
        return

    await ws.accept()
    config = {"configurable": {"thread_id": thread_id}}
    sent_ids: set = set()

    # 断线重连 / 切换会话时，从数据库回放历史消息
    try:
        history = db.execute(
            select(Message).where(Message.conversation_id == conv.id).order_by(Message.id.asc())
        ).scalars().all()
        for m in history:
            sent_ids.add(m.message_id)
            if m.role == "user":
                await ws.send_json({"type": "user", "content": m.content})
            else:
                await ws.send_json({"type": "update", "message": _msg_out(m), "replay": True})

        pending = await _collect_interrupts(config)
        if pending:
            await ws.send_json({"type": "interrupt", "payload": pending})
    except Exception:
        pass

    try:
        while True:
            data = json.loads(await ws.receive_text())
            kind = data.get("kind")

            if kind == "message":
                text = (data.get("text") or "").strip()
                if not text:
                    continue
                user_mid = uuid.uuid4().hex
                _save_message(db, conv.id, user_mid, "user", text)
                _touch_conversation(db, conv, text)
                await ws.send_json({"type": "user", "content": text})
                await ws.send_json({"type": "conversation", "conversation": _conv_out(conv)})

                try:
                    await _stream_and_forward(
                        {"messages": [HumanMessage(content=text)]},
                        config, ws, sent_ids, db, conv.id,
                    )
                except Exception as e:
                    await ws.send_json({"type": "error", "content": f"执行出错: {e}"})
                    continue

                _touch_conversation(db, conv)

                pending = await _collect_interrupts(config)
                if pending:
                    await ws.send_json({"type": "interrupt", "payload": pending})
                else:
                    await ws.send_json({"type": "done"})

            elif kind == "resume":
                try:
                    await _stream_and_forward(
                        Command(resume=data.get("value") or {}),
                        config, ws, sent_ids, db, conv.id,
                    )
                except Exception as e:
                    await ws.send_json({"type": "error", "content": f"恢复出错: {e}"})
                    continue

                _touch_conversation(db, conv)

                pending = await _collect_interrupts(config)
                if pending:
                    await ws.send_json({"type": "interrupt", "payload": pending})
                else:
                    await ws.send_json({"type": "done"})

    except WebSocketDisconnect:
        return
    finally:
        db.close()


# ---------- 静态页 ----------
@app.get("/")
async def index():
    return HTMLResponse(Path("static/index.html").read_text(encoding="utf-8"))
