import json
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.types import Command, Interrupt

from agent_core import parent_graph   # 导入时即完成模型 / 技能 / 子图初始化

app = FastAPI(title="生态调查 Agent")


# ---------- 序列化 ----------
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
        tool_calls = [{"name": tc["name"], "args": tc["args"]} for tc in msg.tool_calls]
    return {"role": role, "content": content, "tool_calls": tool_calls}


def _collect_interrupts(config):
    """读取当前挂起的 interrupt（人工审核 / 参数缺失 / 执行出错）"""
    state = parent_graph.get_state(config)
    out = []
    for task in state.tasks:
        if hasattr(task, "interrupts") and task.interrupts:
            for intr in task.interrupts:
                out.append(intr.value if isinstance(intr, Interrupt) else intr)
    return out


async def _stream_and_forward(payload, config, ws, sent_ids):
    """把图执行过程按消息 id 去重后逐条发给前端"""
    async for event in parent_graph.astream(payload, config=config, stream_mode="values"):
        for msg in event.get("messages", []) or []:
            if isinstance(msg, HumanMessage):
                continue  # 前端已经本地回显
            mid = getattr(msg, "id", None)
            if mid is None or mid in sent_ids:
                continue
            sent_ids.add(mid)
            await ws.send_json({"type": "update", "message": _serialize_message(msg)})


# ---------- WebSocket 主循环 ----------
@app.websocket("/ws/{thread_id}")
async def ws_chat(ws: WebSocket, thread_id: str):
    await ws.accept()
    config = {"configurable": {"thread_id": thread_id}}
    sent_ids: set = set()

    # 断线重连时把历史消息回放一遍
    try:
        state = parent_graph.get_state(config)
        for msg in state.values.get("messages", []) or []:
            if isinstance(msg, HumanMessage):
                await ws.send_json({"type": "user", "content": str(msg.content)})
            else:
                mid = getattr(msg, "id", None)
                if mid:
                    sent_ids.add(mid)
                await ws.send_json({"type": "update", "message": _serialize_message(msg)})
        # 恢复时如果还有挂起的 interrupt
        pending = _collect_interrupts(config)
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
                await ws.send_json({"type": "user", "content": text})
                try:
                    await _stream_and_forward(
                        {"messages": [HumanMessage(content=text)]},
                        config, ws, sent_ids,
                    )
                except Exception as e:
                    await ws.send_json({"type": "error", "content": f"执行出错: {e}"})
                    continue

                pending = _collect_interrupts(config)
                if pending:
                    await ws.send_json({"type": "interrupt", "payload": pending})
                else:
                    await ws.send_json({"type": "done"})

            elif kind == "resume":
                try:
                    await _stream_and_forward(
                        Command(resume=data.get("value") or {}),
                        config, ws, sent_ids,
                    )
                except Exception as e:
                    await ws.send_json({"type": "error", "content": f"恢复出错: {e}"})
                    continue

                pending = _collect_interrupts(config)
                if pending:
                    await ws.send_json({"type": "interrupt", "payload": pending})
                else:
                    await ws.send_json({"type": "done"})

    except WebSocketDisconnect:
        return


# ---------- 静态页 ----------
@app.get("/")
async def index():
    return HTMLResponse(Path("static/index.html").read_text(encoding="utf-8"))