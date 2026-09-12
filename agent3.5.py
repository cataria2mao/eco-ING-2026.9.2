import os, json, re, math
from pathlib import Path
from typing import TypedDict, Optional, Any, List, Dict, Annotated
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, Interrupt
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage, HumanMessage
from langchain_deepseek import ChatDeepSeek

# 子图实现（动物 / 植物 base work agent、技能发现、脚本执行工具等）
# 已拆分到 subgraph_agent.py，便于独立维护与更新
from subgraph_agent import (
    AnimalAnalysisState,
    PlantAnalysisState,
    build_base_work_agent,
    discover_skills,
    _parse_json_safely,
    ANIMAL_SKILLS_ROOT,
    PLANT_SKILLS_ROOT,
)


# ============================================================
# 架构说明（agent3.5）
# ------------------------------------------------------------
# 1. 父图【不承担“规划器”功能】。父图只负责：
#      接收用户消息 → 【需求判断】（动物任务 / 植物任务 / 知识问答）
#      → 数据分析/报告任务：将【原始用户输入原样】转发给对应子图 → 接收子图结果汇报；
#      → 知识问答：从多个知识库中选择最匹配的一个做混合检索 → 基于检索资料简述回答。
# 2. 建立两个领域子图（base work agent）：
#      animal_base_work_agent（动物，状态 AnimalAnalysisState）
#      plant_base_work_agent  （植物，状态 PlantAnalysisState）
#    规划器下沉到子图：子图收到任务后，先由【任务规划器】根据用户输入 + 本子图
#    技能目录输出一个 JSON 决定如何执行（workflow / single_skill / chat）。
# 3. 状态分离：
#      - 子图内部执行字段（execution_mode / workflow_steps / step_outputs ...）
#        与父图完全隔离（父图状态里没有这些键，子图内部通道写在各自 checkpoint
#        命名空间下）。两个子图的内部字段命名一致（“命名空间可以相同”），因为
#        它们各自独立编译、命名空间隔离，互不干扰。
#      - 父图与子图之间只通过“桥接通道”交换最小信息：
#            animal_request / animal_result  （动物）
#            plant_request  / plant_result   （植物）
#        因此 animal_base_work_agent 的结果落在 animal_result，
#        plant_base_work_agent 的结果落在 plant_result，互不覆盖。
# ============================================================


class ParentState(TypedDict):
    """父图状态：只有对话 + 检索 + 路由 + 结果桥接，不含任何子图执行字段。"""
    messages: Annotated[list, add_messages]

    # 需求判断结果（不规划任何执行细节）
    route: Optional[str]          # "animal" | "plant" | "chat"
    route_reason: Optional[str]
    kb: Optional[str]             # route=chat 时选定的知识库 key；"none" 表示无需检索

    # 知识库检索结果（来自选定的知识库）
    retrieved_docs: Optional[List[str]]

    # 桥接通道：分发任务 / 收取结果（对应两个子图，互不干扰）
    animal_request: Optional[str]
    animal_result: Optional[Dict[str, Any]]
    plant_request: Optional[str]
    plant_result: Optional[Dict[str, Any]]


# ========== model ==========
load_dotenv(override=True)

llm = ChatDeepSeek(
    model="deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
    extra_body={"thinking": {"type": "disabled"}}
)


# ========== skills 发现（技能 JSON 位于 subgraph_agent.py 约定的目录） ==========
animal_skill_registry = discover_skills(ANIMAL_SKILLS_ROOT)
plant_skill_registry = discover_skills(PLANT_SKILLS_ROOT)

print(f"[SKILLS] 动物技能 {len(animal_skill_registry)} 个：{[s['id'] for s in animal_skill_registry]}")
print(f"[SKILLS] 植物技能 {len(plant_skill_registry)} 个：{[s['id'] for s in plant_skill_registry]}")


# ========== 编译两个子图（实现见 subgraph_agent.py） ==========
checkpointer = MemorySaver()

animal_base_work_agent = build_base_work_agent(
    request_key="animal_request",
    result_key="animal_result",
    domain="animal",
    domain_label="陆生动物",
    state_cls=AnimalAnalysisState,
    registry=animal_skill_registry,
    planner_llm=llm,
    checkpointer=checkpointer,
)

plant_base_work_agent = build_base_work_agent(
    request_key="plant_request",
    result_key="plant_result",
    domain="plant",
    domain_label="陆生植物",
    state_cls=PlantAnalysisState,
    registry=plant_skill_registry,
    planner_llm=llm,
    checkpointer=checkpointer,
)

print("[GRAPH] animal_base_work_agent / plant_base_work_agent 已编译")


# ============================================================
# 父图：需求判断（不规划）→ 分发子图 或 检索知识库 → 结果汇报
# ============================================================

SOP_DOCX_PATH = os.getenv("SOP_DOCX_PATH", r"D:\PythonProject1\生态环境调查报告工作sop.docx")
CHROMA_DIR = os.getenv("CHROMA_DIR", str(Path(__file__).resolve().parent / "chroma_sop_db"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-v3")
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
EMBED_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")


# ========== 多知识库（多向量数据库）注册表 ==========
# 父图先判断用户需求：
#   - 数据分析/报告任务 → 分发给对应子图（animal / plant）；
#   - 知识问答/科普 → 从下列知识库中选择最匹配的一个进行混合检索并简述回答。
# 每个知识库独立使用一个 chromadb 持久化目录 + collection，并缓存 chunks 供 BM25 使用。
KNOWLEDGE_BASES: Dict[str, Dict[str, Any]] = {
    "sop": {
        "label": "生态环境调查 SOP",
        "description": "生态环境调查工作方案、调查流程、报告撰写规范等",
        "chroma_dir": CHROMA_DIR,
        "collection": "sop_semantic",
        "chunks_file": "sop_chunks.json",
    },
    # 预留：按需新增其它向量数据库，路由会自动识别（无需改图结构）
    # "species": {
    #     "label": "物种知识库",
    #     "description": "陆生动物/植物物种名录、保护级别、居留型、生活习性等",
    #     "chroma_dir": str(Path(__file__).resolve().parent / "chroma_species_db"),
    #     "collection": "species_semantic",
    #     "chunks_file": "species_chunks.json",
    # },
}
DEFAULT_KB = "sop"


def _kb_catalog_text() -> str:
    """知识库目录文本（供需求判断节点选择）。"""
    lines = []
    for key, cfg in KNOWLEDGE_BASES.items():
        lines.append(f'- "{key}"：{cfg.get("label", key)}（{cfg.get("description", "")}）')
    return "\n".join(lines) or "（暂无知识库）"


class _DashScopeEmbeddingFunction:
    """DashScope text-embedding 语义向量化（实现 chromadb embedding_function 协议）。"""

    def __init__(self, model: str = EMBED_MODEL, api_key: str = "", base_url: str = EMBED_BASE_URL):
        self.model = model
        self.api_key = api_key or EMBED_API_KEY
        self.base_url = base_url

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        import requests
        out = []
        for i in range(0, len(texts), 10):
            batch = texts[i:i + 10]
            r = requests.post(
                f"{self.base_url}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={"model": self.model, "input": batch},
                timeout=60,
            )
            r.raise_for_status()
            data = r.json().get("data", [])
            data_sorted = sorted(data, key=lambda x: x.get("index", 0))
            out.extend(d["embedding"] for d in data_sorted)
        return out

    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        return self._embed_batch(list(input))

    def embed_query(self, input):
        return self.__call__(input)

    def name(self) -> str:
        return "dashscope_text_embedding_v3"


class _BM25KeywordIndex:
    """纯 Python Okapi BM25 关键词检索（中文按整词 + 字符 bigram 分词）。"""

    def __init__(self, chunks: List[str]):
        self.chunks = chunks
        self.corpus = [self._tokenize(c) for c in chunks]
        self.n = len(self.corpus)
        self.doc_len = [len(d) for d in self.corpus]
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 0.0
        self.k1 = 1.5
        self.b = 0.75
        self.df: Dict[str, int] = {}
        for doc in self.corpus:
            for t in set(doc):
                self.df[t] = self.df.get(t, 0) + 1
        self.idf = {t: math.log((self.n - df + 0.5) / (df + 0.5) + 1.0) for t, df in self.df.items()}

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        text = (text or "").lower()
        tokens: List[str] = []
        for m in re.findall(r'[a-z0-9_]+', text):
            tokens.append(m)
        for m in re.findall(r'[\u4e00-\u9fff]+', text):
            tokens.append(m)
            for i in range(len(m) - 1):
                tokens.append(m[i:i + 2])
        return tokens

    def _score(self, query_tokens: List[str], doc_idx: int) -> float:
        doc = self.corpus[doc_idx]
        tf: Dict[str, int] = {}
        for t in doc:
            tf[t] = tf.get(t, 0) + 1
        dl = self.doc_len[doc_idx]
        score = 0.0
        for t in set(query_tokens):
            if t not in tf:
                continue
            idf = self.idf.get(t, 0.0)
            f = tf[t]
            denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += idf * f * (self.k1 + 1) / denom
        return score

    def search(self, query: str, k: int = 4) -> List[int]:
        if self.n == 0:
            return []
        qt = self._tokenize(query)
        scores = [self._score(qt, i) for i in range(self.n)]
        ranked = sorted(range(self.n), key=lambda i: scores[i], reverse=True)
        return [i for i in ranked if scores[i] > 0][:k]


def _rrf_fuse(rank_lists: List[List[int]], k: int = 60) -> List[int]:
    """Reciprocal Rank Fusion 融合多个排序列表。"""
    scores: Dict[int, float] = {}
    for rl in rank_lists:
        for rank, idx in enumerate(rl):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda i: scores[i], reverse=True)


_kb_chunks_cache: Dict[str, List[str]] = {}
_kb_semantic_store: Dict[str, Any] = {}
_kb_keyword_index: Dict[str, Any] = {}


def _kb_config(kb: str) -> Optional[Dict[str, Any]]:
    return KNOWLEDGE_BASES.get(kb)


def _load_chunks(kb: str = DEFAULT_KB) -> List[str]:
    """加载指定知识库的文本块（优先读缓存文件，否则从 chromadb 读取）。"""
    if kb in _kb_chunks_cache:
        return _kb_chunks_cache[kb]
    cfg = _kb_config(kb)
    if not cfg:
        print(f"[VEC] 未知知识库: {kb}")
        return []
    chroma_dir = cfg.get("chroma_dir", CHROMA_DIR)
    chunks: List[str] = []
    cache_file = Path(chroma_dir) / cfg.get("chunks_file", "chunks.json")
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
    else:
        try:
            import chromadb
            client = chromadb.PersistentClient(path=chroma_dir)
            coll = client.get_collection(cfg.get("collection", "sop_semantic"))
            data = coll.get(include=["documents"])
            chunks = data.get("documents", []) or []
        except Exception as e:
            print(f"[VEC] 无法加载知识库[{kb}] chunks: {e}")
            chunks = []
    _kb_chunks_cache[kb] = chunks
    return chunks


def _get_retrievers(kb: str = DEFAULT_KB):
    """获取指定知识库的语义检索器与关键词检索器（惰性初始化，按知识库隔离）。"""
    chunks = _load_chunks(kb)
    cfg = _kb_config(kb) or {}

    if kb not in _kb_keyword_index:
        _kb_keyword_index[kb] = _BM25KeywordIndex(chunks)

    if kb not in _kb_semantic_store:
        try:
            import chromadb
            client = chromadb.PersistentClient(path=cfg.get("chroma_dir", CHROMA_DIR))
            _kb_semantic_store[kb] = client.get_collection(
                name=cfg.get("collection", "sop_semantic"),
                embedding_function=_DashScopeEmbeddingFunction(),
            )
            print(f"[VEC] 已加载知识库[{kb}]语义索引（{len(chunks)} 块）")
        except Exception as e:
            print(f"[VEC] 知识库[{kb}]语义索引加载失败，仅使用关键词检索: {e}")
            _kb_semantic_store[kb] = None

    return _kb_semantic_store[kb], _kb_keyword_index[kb]


def retrieve_kb_chunks(kb: str, query: str, k: int = 5) -> List[str]:
    """在指定知识库中混合检索：语义（DashScope）+ 关键词（BM25），RRF 融合。"""
    chunks = _load_chunks(kb)
    if not chunks:
        return []
    sem, kw = _get_retrievers(kb)
    idx_of = {c: i for i, c in enumerate(chunks)}
    n = max(k, 3)

    sem_rank: List[int] = []
    if sem is not None:
        try:
            res = sem.query(query_texts=[query], n_results=min(len(chunks), n))
            for d in (res.get("documents") or [[]])[0]:
                i = idx_of.get(d)
                if i is not None and i not in sem_rank:
                    sem_rank.append(i)
        except Exception as e:
            print(f"[VEC] 知识库[{kb}]语义检索失败: {e}")

    kw_rank = kw.search(query, n)
    fused = _rrf_fuse([sem_rank, kw_rank])
    return [chunks[i] for i in fused[:k]]


# ========== 父图 LLM 与工具 ==========

parent_llm = ChatDeepSeek(
    model="deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
    extra_body={"thinking": {"type": "disabled"}}
)


@tool
def parent_list_files(directory: str = ".") -> List[str]:
    """列出指定目录下的所有文件。用于父图协调层查看工作目录内容。"""
    try:
        return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    except Exception as e:
        return [f"错误：{e}"]


@tool
def parent_read_file_content(file_path: str) -> str:
    """读取文本文件（.txt/.py/.json/.csv 等）内容。不支持二进制文件。"""
    try:
        if not os.path.exists(file_path):
            return f"错误：文件 {file_path} 不存在"
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()[:5000]
    except Exception as e:
        return f"读取失败：{e}"


parent_tools = [parent_list_files, parent_read_file_content]
parent_tool_node = ToolNode(parent_tools)
parent_llm_with_tools = parent_llm.bind_tools(parent_tools)


# ---------- 需求判断系统提示（先判需求，再选处理方式） ----------
PARENT_ROUTER_SYSTEM_PROMPT = """你是生态环境调查系统的"需求判断与任务路由"，只负责判断用户需求并决定处理方式，【不做任何执行规划】。

请先判断用户需求，再输出 JSON：
{"route": "animal" | "plant" | "chat", "kb": "<知识库key或none>", "reason": "..."}

route 判定：
- "animal"：用户要求的是【动物类数据分析/报告撰写/名录处理任务】（涉及动物调查数据文件，如鸟类/兽类/两栖爬行监测样线、动物名录、居留型、动物多样性统计、动物报告撰写等，需要执行数据分析脚本完成任务）。
- "plant"：同理，植物类数据分析/报告撰写任务。
- "chat"：一般知识问答（例如询问某个物种的生活习性、生态学概念解释）、闲聊、查看文件、与数据分析脚本执行无关的请求。

kb 判定（仅 route=chat 时有效）：
- 若问题需要知识库资料支撑，从下列可用知识库中选择最匹配的一个，kb 填其 key；
- 闲聊、问候、查看目录/文件等无需检索，kb 填 "none"；
- route=animal/plant 时 kb 一律填 "none"。

可用知识库：
{kb_catalog}

注意：
1. 询问物种习性、科普知识等即使提到动物/植物名，也归入 "chat"，并选择相应知识库。
2. 只有"需要处理数据文件并执行分析/报告任务"的请求才路由给子图（animal/plant）。
3. 只输出 JSON，不要输出其他文字。"""


# ---------- 父图汇报/问答系统提示 ----------
PARENT_SYSTEM_PROMPT = """你是生态环境调查（陆生动物 / 陆生植物）项目的协调助手。不负责规划与执行：数据分析、报告撰写任务会原样转交给对应的领域子图（animal_base_work_agent / plant_base_work_agent）去规划并执行。你负责：

1. 一般问答：
   - 若检索命中了与问题相关的知识库资料，请【基于检索资料简要回答】，并可简要说明资料要点及来源知识库；
   - 若所选知识库未命中任何相关内容，而问题需要专业知识/资料支撑（如询问某物种的生活习性、特定方法细节、知识库收录内容等），请明确告知用户"当前知识库暂未收录相关内容"，不要编造答案；
   - 寒暄、闲聊可以正常回应。

2. 用户要求查看目录/读取文件时，使用 parent_list_files / parent_read_file_content 工具。

3. 数据分析/报告任务：子图执行完成后，系统会给出子图执行结果（SystemMessage），你负责据此向用户清晰、友好地汇报（任务状态、产物文件等）。

严禁自行模拟或编造数据分析执行结果；执行结果以系统消息为准。"""


def _get_last_human_text(state) -> str:
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            return str(m.content)
    return ""


# ---------- 父图节点 ----------

def parent_router_node(state: ParentState) -> Dict:
    """需求判断节点：先判断用户需求，再决定【分发子图】或【检索知识库】。"""
    user_text = _get_last_human_text(state)
    system_prompt = PARENT_ROUTER_SYSTEM_PROMPT.replace("{kb_catalog}", _kb_catalog_text())

    resp = parent_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"用户输入：\n{user_text}\n\n请输出 JSON："),
    ])
    content = resp.content if hasattr(resp, 'content') else str(resp)
    decision = _parse_json_safely(content)

    route = decision.get("route")
    if route not in ("animal", "plant"):
        route = "chat"

    kb = decision.get("kb")
    if route in ("animal", "plant") or kb not in KNOWLEDGE_BASES:
        kb = "none"

    print(f"[DEBUG] parent_router: route={route}, kb={kb}, reason={decision.get('reason', '')}")
    return {
        "route": route,
        "route_reason": decision.get("reason", ""),
        "kb": kb,
        "retrieved_docs": [],   # 清空上一轮检索结果，避免串味
    }


def parent_router_router(state: ParentState) -> str:
    """需求判断后走向：数据分析→分发子图；知识问答→检索所选知识库；其余→直接回答。"""
    if state.get("route") in ("animal", "plant"):
        return "dispatch"
    if state.get("kb") in KNOWLEDGE_BASES:
        return "retrieve"
    return "chat"


def parent_retrieve_node(state: ParentState) -> Dict:
    """检索节点：在需求判断选定的知识库中做混合检索。"""
    kb = state.get("kb") or "none"
    user_message = _get_last_human_text(state)
    if kb not in KNOWLEDGE_BASES:
        return {"retrieved_docs": []}
    docs = retrieve_kb_chunks(kb, user_message, k=5)
    label = KNOWLEDGE_BASES[kb].get("label", kb)
    print(f"[DEBUG] parent_retrieve: 知识库[{kb}/{label}] 检索到 {len(docs)} 条资料" + ("" if docs else "（无）"))
    return {"retrieved_docs": docs}


def parent_dispatch_node(state: ParentState) -> Dict:
    """分发节点：把【原始用户输入】（不附加任何规划内容）写入对应子图的桥接通道。"""
    user_text = _get_last_human_text(state)
    route = state.get("route")
    update: Dict = {}
    if route == "animal":
        update = {"animal_request": user_text, "animal_result": None,
                  "plant_request": "", "plant_result": None}
    elif route == "plant":
        update = {"plant_request": user_text, "plant_result": None,
                  "animal_request": "", "animal_result": None}
    print(f"[DEBUG] parent_dispatch: 将用户输入原样转发至 {route}_base_work_agent")
    return update


def parent_dispatch_router(state: ParentState) -> str:
    """按路由选择执行哪个子图。"""
    if state.get("route") == "animal":
        return "animal_base_work_agent"
    if state.get("route") == "plant":
        return "plant_base_work_agent"
    return "parent_chat"


def _format_domain_result(domain: str, domain_label: str, result: Optional[Dict]) -> Optional[str]:
    """把子图返回的桥接结果格式化为汇报文本。"""
    if not result:
        return None
    fo = result.get("final_output") if isinstance(result, dict) else result
    if not isinstance(fo, dict):
        fo = {"status": "unknown"}
    status = fo.get("status", "unknown")
    skill_name = fo.get("skill", "未知任务")
    req = (result.get("request") or "")[:80] if isinstance(result, dict) else ""

    if status == "completed":
        msg = f"✅ {domain_label}子图任务『{skill_name}』执行完成。\n"
        if fo.get("type") == "single":
            for r in fo.get("results", []):
                msg += f"  - {r.get('description', r.get('script', ''))}\n"
        else:
            for key, val in (fo.get("step_outputs") or {}).items():
                if isinstance(val, dict) and val.get("type") != "review":
                    files = val.get("output_files", [])
                    if files:
                        msg += f"  {key}: 输出文件 {files}\n"
    elif status == "error":
        msg = f"❌ {domain_label}子图任务『{skill_name}』执行出错：\n{fo.get('error', '未知错误')}"
    elif status == "aborted":
        msg = f"⚠️ {domain_label}子图任务『{skill_name}』已终止：{fo.get('reason', '人工终止')}"
    elif status == "need_info":
        msg = f"❓ {domain_label}子图规划器需要补充信息：{fo.get('reason', '')}"
    else:
        msg = f"{domain_label}子图任务『{skill_name}』状态：{status}"
    if req:
        msg = f"（原始任务：{req}）\n{msg}"
    return msg


def parent_report_node(state: ParentState) -> Dict:
    """汇合节点：子图执行完成后，汇总 animal_result / plant_result 并交回对话层汇报。"""
    msgs = []
    if state.get("animal_result"):
        txt = _format_domain_result("animal", "陆生动物", state["animal_result"])
        if txt:
            print(f"[DEBUG] parent_report(animal): {txt[:200]}")
            msgs.append(SystemMessage(content=txt))
    if state.get("plant_result"):
        txt = _format_domain_result("plant", "陆生植物", state["plant_result"])
        if txt:
            print(f"[DEBUG] parent_report(plant): {txt[:200]}")
            msgs.append(SystemMessage(content=txt))

    update: Dict = {
        "animal_request": "",
        "plant_request": "",
        "route": "chat",   # 本次委派结束，路由复位
        "kb": "none",      # 清空知识库选择
    }
    # 注意：animal_result / plant_result 保留在父图状态中，供后续查看/汇报使用；
    # 下一次委派时 parent_dispatch_node 会先将其清空，因此不会串味。
    if msgs:
        update["messages"] = msgs
    return update


def parent_chat_node(state: ParentState):
    """父图聊天节点：回答一般问题 / 查看文件 / 汇报子图执行结果。"""
    docs = state.get("retrieved_docs") or []
    kb = state.get("kb")
    kb_label = KNOWLEDGE_BASES.get(kb, {}).get("label", "") if kb else ""
    if docs:
        source = f"（来源知识库：{kb_label}）" if kb_label else ""
        doc_block = f"\n\n检索到的相关资料{source}：\n" + "\n\n".join(
            f"[资料{i + 1}] {d[:800]}" for i, d in enumerate(docs))
    elif kb_label:
        doc_block = f"\n\n检索到的相关资料（来源知识库：{kb_label}）：（无）"
    else:
        doc_block = ""
    messages = [SystemMessage(content=PARENT_SYSTEM_PROMPT + doc_block)] + state["messages"]
    response = parent_llm_with_tools.invoke(messages)

    if hasattr(response, 'tool_calls') and response.tool_calls:
        tc_summary = ", ".join(
            f"{tc['name']}({json.dumps(tc['args'], ensure_ascii=False)[:100]})" for tc in response.tool_calls)
        print(f"[DEBUG] parent_chat: LLM 调用工具 → {tc_summary}")
    elif response.content:
        print(f"[DEBUG] parent_chat: LLM 回复 → {response.content[:200]}")

    return {"messages": [response]}


def parent_chat_router(state: ParentState) -> str:
    last_message = state["messages"][-1] if state["messages"] else None
    if isinstance(last_message, AIMessage) and getattr(last_message, 'tool_calls', None):
        return "tools"
    return "respond"


# ---------- 构建父图 ----------

parent_builder = StateGraph(ParentState)

parent_builder.add_node("parent_retrieve", parent_retrieve_node)
parent_builder.add_node("parent_router", parent_router_node)
parent_builder.add_node("parent_dispatch", parent_dispatch_node)
parent_builder.add_node("parent_report", parent_report_node)
parent_builder.add_node("parent_chat", parent_chat_node)
parent_builder.add_node("parent_tools", parent_tool_node)
parent_builder.add_node("animal_base_work_agent", animal_base_work_agent)
parent_builder.add_node("plant_base_work_agent", plant_base_work_agent)

# 需求判断（入口）
parent_builder.add_edge(START, "parent_router")

# 需求判断 → 分发子图 / 检索知识库 / 直接回答
parent_builder.add_conditional_edges("parent_router", parent_router_router, {
    "dispatch": "parent_dispatch",
    "retrieve": "parent_retrieve",
    "chat": "parent_chat",
})

# 知识库检索 → 对话层简述回答
parent_builder.add_edge("parent_retrieve", "parent_chat")

# 分发 → 对应子图（一次只委派一个领域）
parent_builder.add_conditional_edges("parent_dispatch", parent_dispatch_router, {
    "animal_base_work_agent": "animal_base_work_agent",
    "plant_base_work_agent": "plant_base_work_agent",
    "parent_chat": "parent_chat",
})

# 子图 → 汇总汇报 → 对话层
parent_builder.add_edge("animal_base_work_agent", "parent_report")
parent_builder.add_edge("plant_base_work_agent", "parent_report")
parent_builder.add_edge("parent_report", "parent_chat")

# 对话层（工具循环）
parent_builder.add_conditional_edges("parent_chat", parent_chat_router, {
    "tools": "parent_tools",
    "respond": END,
})
parent_builder.add_edge("parent_tools", "parent_chat")

parent_graph = parent_builder.compile(checkpointer=checkpointer)

print("[GRAPH] 父图已编译：需求判断 → (animal/plant 子图 或 知识库检索) → 汇报")


# ========== 主入口测试 ==========
def _print_last_ai_message(agent, config):
    """输出最后一条 AI 消息"""
    state = agent.get_state(config)
    messages = state.values.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            print(f"\n🤖 助手: {msg.content}")
            return
    for key in ("animal_result", "plant_result"):
        res = state.values.get(key)
        if res:
            print(f"\n🤖 子图结果: {json.dumps(res, ensure_ascii=False, indent=2, default=str)}")
            return


def _handle_interrupts(agent, config):
    """检查并处理图中的 interrupt 状态（人工审核/失败中断）"""
    state = agent.get_state(config)
    for task in state.tasks:
        if not (hasattr(task, 'interrupts') and task.interrupts):
            continue

        for intr in task.interrupts:
            interrupt_value = intr.value if isinstance(intr, Interrupt) else intr
            print(f"\n📋 审核请求: {json.dumps(interrupt_value, ensure_ascii=False, indent=2, default=str)}")

            if interrupt_value.get("type") == "missing_required_params":
                missing = interrupt_value.get("missing_params", [])
                print(f"技能『{interrupt_value.get('skill', '')}』以下必填参数缺失：")
                for item in missing:
                    name = item.get("name") if isinstance(item, dict) else str(item)
                    desc = item.get("description", "") if isinstance(item, dict) else ""
                    print(f"  - {name}：{desc}")
                while True:
                    choice = input("请选择 (提供参数/默认参数): ").strip().lower()
                    if choice in ("提供参数", "提供", "provide", "p"):
                        provided = {}
                        for item in missing:
                            name = item.get("name") if isinstance(item, dict) else str(item)
                            desc = item.get("description", "") if isinstance(item, dict) else ""
                            val = input(f"请输入 {name}（{desc}）[直接回车=使用脚本默认参数]: ").strip()
                            if val:
                                provided[name] = val
                        resume_value = {"action": "provide", "params": provided}
                        break
                    elif choice in ("默认参数", "默认", "default", "d"):
                        resume_value = {"action": "default"}
                        break
                    else:
                        print("无效输入，请输入: 提供参数 或 默认参数")
            elif interrupt_value.get("type") == "execution_error":
                print(f"错误详情: {interrupt_value.get('error')}")
                while True:
                    choice = input("请选择 (重试/终止): ").strip().lower()
                    if choice in ("重试", "retry"):
                        feedback = input("请输入修改建议（可选）: ").strip()
                        resume_value = {"approved": False, "action": "retry", "feedback": feedback}
                        break
                    elif choice in ("终止", "abort"):
                        resume_value = {"approved": False, "action": "abort"}
                        break
                    else:
                        print("无效输入，请输入: 重试 或 终止")
            else:
                while True:
                    action_input = input("\n请选择操作 (通过/重新执行/终止): ").strip()
                    if action_input in ("通过", "approve", "continue"):
                        resume_value = {"approved": True, "action": "continue"}
                        break
                    elif action_input in ("重新执行", "retry"):
                        feedback = input("请输入修改建议（可选）: ").strip()
                        resume_value = {"approved": False, "action": "retry", "feedback": feedback}
                        break
                    elif action_input in ("终止", "abort"):
                        resume_value = {"approved": False, "action": "abort"}
                        break
                    else:
                        print("无效输入，请选择: 通过 / 重新执行 / 终止")

            # 恢复执行
            for _ in agent.stream(Command(resume=resume_value), config=config, stream_mode="values"):
                pass

            # 恢复后可能还有新的 interrupt，递归处理
            _handle_interrupts(agent, config)
            return


def run_interactive():
    """交互式运行"""
    config = {"configurable": {"thread_id": "interactive-1"}}

    print("=" * 60)
    print("生态环境调查数据分析助手（agent3.5）")
    print("父图：需求判断 → (分发子图 / 检索知识库) → 汇报")
    print("子图：animal_base_work_agent（动物）/ plant_base_work_agent（植物），内含任务规划器")
    print("动物技能：", [s["name"] for s in animal_skill_registry])
    print("植物技能：", [s["name"] for s in plant_skill_registry])
    print("输入 'quit' 退出")
    print("=" * 60)

    while True:
        user_input = input("\n🧑 你: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("再见！")
            break
        if not user_input:
            continue

        initial_state = {
            "messages": [HumanMessage(content=user_input)],
        }

        try:
            step_count = 0
            for event in parent_graph.stream(initial_state, config=config, stream_mode="values"):
                step_count += 1
                msgs = event.get("messages", [])
                if msgs:
                    latest = msgs[-1]
                    role = type(latest).__name__
                    if isinstance(latest, HumanMessage):
                        role = "👤 Human"
                    elif isinstance(latest, AIMessage):
                        tc_info = ""
                        if hasattr(latest, 'tool_calls') and latest.tool_calls:
                            tc_names = [tc["name"] for tc in latest.tool_calls]
                            tc_info = f" [工具: {', '.join(tc_names)}]"
                        role = f"🤖 AI{tc_info}"
                    elif isinstance(latest, ToolMessage):
                        role = "🔧 Tool"
                    elif isinstance(latest, SystemMessage):
                        role = "📋 System"
                    content = latest.content or ""
                    print(f"  [STEP {step_count}] {role}: {content}")

            # 检查是否有 pending interrupt（人工审核）
            _handle_interrupts(parent_graph, config)

            # 输出最后一条 AI 消息
            _print_last_ai_message(parent_graph, config)

        except Exception as e:
            print(f"\n❌ 执行出错: {e}")


if __name__ == "__main__":
    run_interactive()
