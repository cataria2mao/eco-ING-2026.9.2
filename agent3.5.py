import os, json, subprocess, re, sys, math
from datetime import datetime
from docx import Document
from pathlib import Path
from typing import TypedDict, Optional, Any, List, Dict, Annotated
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt, Command, Interrupt
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage, HumanMessage
from langchain_deepseek import ChatDeepSeek
from langchain_docling.loader import DoclingLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# 架构说明（agent3.5）
# ------------------------------------------------------------
# 1. 父图【不承担“规划器”功能】。父图只负责：
#      混合检索 SOP 向量数据库 → 判断用户请求归属（动物任务/植物任务/直接问答）
#      → 将【原始用户输入原样】转发给对应子图 → 接收子图结果并向用户汇报。
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


# ========== 状态定义 ==========

class AnimalAnalysisState(TypedDict):
    """动物子图 animal_base_work_agent 的状态（与父图状态分离）。"""
    # --- 桥接通道：仅这两个键与父图同名共享（随任务置位、结束后清空） ---
    animal_request: Optional[str]                 # 父图转发的原始用户任务文本
    animal_result: Optional[Dict[str, Any]]       # 子图返回父图的最终结果
    # --- 共享对话通道（父图/子图可见；子图执行路径不写 messages） ---
    messages: Annotated[list, add_messages]
    # --- 内部执行字段（父图不可见，不写入父图） ---
    domain: Optional[str]                         # "animal"
    execution_mode: str                           # "chat"|"workflow"|"single_skill"|"need_info"|"idle"
    skill_registry: Optional[List[Dict]]
    selected_skill: Optional[str]
    skill_config: Optional[Dict]
    skill_params: Optional[Dict]
    work_dir: str
    input_file: str
    history_file: str
    pa: str
    regional_level: str
    current_step_idx: int
    workflow_steps: List[Dict]
    step_outputs: Dict[str, Any]
    review_feedback: Optional[str]
    approved: Optional[bool]
    review_action: Optional[str]
    retry_count: Dict[str, int]
    retry_target_idx: Optional[int]
    final_output: Optional[Any]
    error: Optional[str]
    sub_delegated: Optional[bool]                 # 本轮是否为父图委派


class PlantAnalysisState(TypedDict):
    """植物子图 plant_base_work_agent 的状态（与父图状态分离）。"""
    # --- 桥接通道：仅这两个键与父图同名共享 ---
    plant_request: Optional[str]
    plant_result: Optional[Dict[str, Any]]
    # --- 共享对话通道 ---
    messages: Annotated[list, add_messages]
    # --- 内部执行字段（与动物子图命名一致，父图不可见） ---
    domain: Optional[str]                         # "plant"
    execution_mode: str
    skill_registry: Optional[List[Dict]]
    selected_skill: Optional[str]
    skill_config: Optional[Dict]
    skill_params: Optional[Dict]
    work_dir: str
    input_file: str
    history_file: str
    pa: str
    regional_level: str
    current_step_idx: int
    workflow_steps: List[Dict]
    step_outputs: Dict[str, Any]
    review_feedback: Optional[str]
    approved: Optional[bool]
    review_action: Optional[str]
    retry_count: Dict[str, int]
    retry_target_idx: Optional[int]
    final_output: Optional[Any]
    error: Optional[str]
    sub_delegated: Optional[bool]


class ParentState(TypedDict):
    """父图状态：只有对话 + 检索 + 路由 + 结果桥接，不含任何子图执行字段。"""
    messages: Annotated[list, add_messages]

    # 混合检索结果
    retrieved_docs: Optional[List[str]]

    # 路由判断结果（不规划任何执行细节）
    route: Optional[str]          # "animal" | "plant" | "chat"
    route_reason: Optional[str]

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


# ========== skills 发现与加载 ==========
def discover_skills(skills_root: Path) -> List[Dict[str, Any]]:
    """扫描 skills_root/*.json，提取 name, description 和 json 路径"""
    skills = []
    if not skills_root.exists():
        return skills
    for json_file in skills_root.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                full_info = json.load(f)
            skill_basic = {
                "name": full_info.get("name", json_file.stem),
                "description": full_info.get("description", ""),
                "id": full_info.get("id", json_file.stem),
                "_json_path": str(json_file)
            }
            skills.append(skill_basic)
        except Exception as e:
            print(f"警告：读取 {json_file} 失败：{e}")
    return skills


def _load_full_skill_from(registry: List[Dict], skill_name: str) -> tuple:
    """从指定技能注册表加载技能的完整配置"""
    skill_basic = None
    for s in registry:
        if s["name"] == skill_name or s.get("id") == skill_name:
            skill_basic = s
            break

    if not skill_basic:
        available = [s["name"] for s in registry]
        return None, f"错误：未找到技能 '{skill_name}'，可用：{', '.join(available)}"

    json_path = skill_basic["_json_path"]
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            full_skill = json.load(f)
    except Exception as e:
        return None, f"错误：无法加载技能配置 {json_path}: {e}"

    return full_skill, json_path


# 动物技能放在 skills/ 根目录；植物技能放在 plant_skills/ 目录（各自独立发现）
# 说明：植物技能暂未创建/验证，plant_base_work_agent 的规划器与执行器已就绪，
# 后续只需在 plant_skills/ 下放入植物技能 JSON（及对应脚本）即可自动启用。
ANIMAL_SKILLS_ROOT = Path("./skills")
PLANT_SKILLS_ROOT = Path("./plant_skills")

animal_skill_registry = discover_skills(ANIMAL_SKILLS_ROOT)
plant_skill_registry = discover_skills(PLANT_SKILLS_ROOT)

print(f"[SKILLS] 动物技能 {len(animal_skill_registry)} 个：{[s['id'] for s in animal_skill_registry]}")
print(f"[SKILLS] 植物技能 {len(plant_skill_registry)} 个：{[s['id'] for s in plant_skill_registry]}")


# ========== 共享工具（脚本执行 / 文件读取） ==========
@tool
def list_files(directory: str = ".") -> List[str]:
    """列出指定目录下的所有文件"""
    try:
        return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    except Exception as e:
        return [f"错误：{e}"]


@tool
def read_xlsx(file_path: str) -> str:
    """读取 Excel 文件 (xlsx) 并返回文本内容"""
    import pandas as pd
    try:
        if not os.path.exists(file_path):
            return f"错误：文件 {file_path} 不存在"
        df = pd.read_excel(file_path, sheet_name=None)
        output = []
        for sheet_name, data in df.items():
            output.append(f"工作表: {sheet_name}\n{data.to_string()}")
        return "\n\n".join(output)
    except Exception as e:
        return f"读取失败：{e}"


@tool
def read_docx(file_path: str) -> str:
    """读取 Word 文件 (docx) 并返回纯文本"""
    try:
        if not os.path.exists(file_path):
            return f"错误：文件 {file_path} 不存在"
        doc = Document(file_path)
        full_text = [para.text for para in doc.paragraphs]
        return "\n".join(full_text)
    except Exception as e:
        return f"读取失败：{e}"


@tool
def run_python(script_path: str, script_args: str) -> str:
    """运行 python 脚本，传递关键词参数。

    Args:
        script_path: Python 脚本路径
        script_args: 命令行参数，空格分隔，如 "--work_dir ./data --input_file test.xlsx"
    """
    import subprocess as sp
    try:
        if not os.path.exists(script_path):
            return f"错误：脚本 {script_path} 不存在"
        sa_list = script_args.split() if script_args.strip() else []
        cmd = [sys.executable, script_path] + sa_list
        result = sp.run(cmd, capture_output=True, text=True, encoding='utf-8', timeout=120)
        if result.returncode != 0:
            return f"执行失败 (code {result.returncode}):\nSTDERR:\n{result.stderr}\nSTDOUT:\n{result.stdout}"
        return result.stdout
    except sp.TimeoutExpired:
        return "错误：脚本执行超时（120秒）"
    except Exception as e:
        return f"执行异常：{e}"


@tool
def run_r(script_path: str, script_args: str) -> str:
    """运行 R 脚本，传递关键词参数。

    Args:
        script_path: R 脚本路径
        script_args: 命令行参数，空格分隔，如 "--work_dir ./data --input_file test.xlsx"
    """
    try:
        if not os.path.exists(script_path):
            return f"错误：脚本 {script_path} 不存在"
        rscript = "Rscript" if os.name != "nt" else "Rscript.exe"
        sa_list = script_args.split() if script_args.strip() else []
        cmd = [rscript, script_path] + sa_list
        result = subprocess.run(cmd, capture_output=True,
                                text=True,
                                encoding='utf-8',
                                errors='replace',
                                timeout=120)
        if result.returncode != 0:
            return f"执行失败 (code {result.returncode}):\nSTDERR:\n{result.stderr}\nSTDOUT:\n{result.stdout}"
        return result.stdout
    except FileNotFoundError:
        return "错误：未找到 Rscript，请确认R已安装并加入PATH"
    except subprocess.TimeoutExpired:
        return "错误：脚本执行超时（120秒）"
    except Exception as e:
        return f"执行异常：{e}"


# ========== 纯辅助函数（领域无关） ==========

def detect_step_type(step: Dict) -> str:
    """统一检测步骤类型"""
    if step.get("use_skill") == "review":
        return "review"
    if step.get("review_point") is True and "use_skill" not in step:
        return "review"
    if step.get("type") == "review":
        return "review"
    return "skill"


def parse_review_action(response: Dict) -> str:
    """解析 review 响应为统一 action"""
    if response.get("approved") is True:
        return "continue"
    action = response.get("action", "")
    if action in ("continue", "通过", "approve"):
        return "continue"
    elif action in ("retry", "重新执行", "修改", "retry_previous"):
        return "retry"
    else:
        return "abort"


def extract_output_files(results: List[Dict], params: Dict) -> List[str]:
    files = []
    work_dir = Path(params.get("work_dir", ".")).resolve()
    if "output_file" in params:
        files.append(str(work_dir / params["output_file"]))
    for r in results:
        stdout = r.get("result", "")
        for line in stdout.split("\n"):
            for ext in [".xlsx", ".csv", ".txt", ".docx"]:
                if ext in line:
                    parts = line.split()
                    for part in parts:
                        if part.endswith(ext):
                            files.append(str(Path(part).resolve()))
    return list(set(files))


def _replace_vars(obj, params_vars: Dict[str, Any]):
    """递归替换模板变量"""
    if isinstance(obj, str):
        if '{' not in obj:
            return obj
        parts = re.split(r'(\{[^{}]+\})', obj)
        result_parts = []
        has_missing = False
        for part in parts:
            if part.startswith('{') and part.endswith('}'):
                key = part[1:-1].strip()
                if key in params_vars:
                    val = params_vars[key]
                    result_parts.append(str(val) if val is not None else "")
                else:
                    has_missing = True
            else:
                result_parts.append(part)
        if has_missing:
            return None
        return ''.join(result_parts)
    elif isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            new_v = _replace_vars(v, params_vars)
            if new_v is not None:
                new_dict[k] = new_v
        return new_dict
    elif isinstance(obj, list):
        new_list = []
        for item in obj:
            new_item = _replace_vars(item, params_vars)
            if new_item is not None:
                new_list.append(new_item)
        return new_list
    else:
        return obj


def substitute_params(script_args: List[str], params: Dict) -> List[str]:
    """将 script_args 中的 {var} 替换为 params 中的值"""
    result = []
    for sa in script_args:
        for key, val in params.items():
            sa = sa.replace(f"{{{key}}}", str(val))
        result.append(sa)
    return result


def _clean_unresolved_script_args(script_args: List[str]) -> List[str]:
    """成对移除未替换的可选参数：如 "--ref_file {ref_file}" → 两个都移除"""
    cleaned = []
    for sa in script_args:
        if '{' in sa and '}' in sa:
            if cleaned and cleaned[-1].startswith('--'):
                cleaned.pop()
            continue
        cleaned.append(sa)
    return cleaned


def _parse_json_safely(content: str) -> Dict:
    if not content:
        return {}
    content = content.strip()
    m = re.search(r'\{[\s\S]*\}', content)
    if m:
        content = m.group(0)
    try:
        return json.loads(content)
    except Exception:
        try:
            return json.loads(content.replace('，', ','))
        except Exception:
            return {}


def _normalize_workflow_steps(steps: List[Dict]) -> List[Dict]:
    """规整工作流步骤：移除 input_from 为 None/0/空字符串的依赖标记"""
    clean = []
    for st in steps:
        s = dict(st)
        if s.get("input_from") in (None, 0, ""):
            s.pop("input_from", None)
        clean.append(s)
    return clean


# ============================================================
# 子图构造器（动物 / 植物共用一套内部结构，命名空间一致）
# ------------------------------------------------------------
# 每个子图：
#   父图委派(REQUEST 桥接) → 任务规划器(输出 JSON) → executor / single_executor
#     →（审核 / 失败中断）→ finish 节点把结果写回 RESULT 桥接通道
# ============================================================

SUB_PLANNER_SYSTEM_PROMPT = """你是{domain_label}数据分析子图的"任务规划器"。
你只接收父图转发的【原始用户任务文本】（不附带任何预规划内容）。你的职责是：
根据用户输入 + 本子图可用技能目录，匹配相关技能并【输出一个 JSON】决定如何执行。

【可用技能目录】（见用户消息，每行一个技能 JSON）。规划时必须遵守：
- use_skill / skill_id 只能使用技能目录中的 id 或 name。
- workflow_steps 每项形如：{{"step": 1, "type": "skill", "use_skill": "<id>", "input_from": <依赖的上一步 step 编号>}}；
  审核步骤：{{"step": n, "type": "review", "description": "..."}}。没有依赖时可省略 input_from。
- 用户说"全工作流 / 完整流程 / 报告撰写"时，优先复用目录中 type=workflow 技能的 workflow_steps；
  否则按用户意图把多个 single 技能按顺序拼接成 workflow_steps。
- 用户只要求单一技能时，用 action=single_skill。
- parameters 只填用户明确给出的参数（work_dir / input_file / history_file / pa / regional_level 等），
  用户未提供的键省略（脚本有默认值，不要编造文件路径）。
- 无法确定要执行什么、缺关键信息，或用户输入属于一般问答/闲聊（非数据分析任务）时，
  输出 {{"action": "chat", "reason": "需要向用户澄清/说明的内容"}}。

输出 JSON 三选一：
{{"action": "workflow", "mode": "workflow", "task_name": "...", "skill_id": "...", "parameters": {{...}}, "workflow_steps": [...]}}
或
{{"action": "single_skill", "mode": "single_skill", "task_name": "...", "skill_id": "...", "parameters": {{...}}}}
或
{{"action": "chat", "reason": "..."}}

只输出 JSON，不要输出其他文字。"""


def _build_skill_catalog_text(registry: List[Dict]) -> str:
    lines = []
    for s in registry:
        cfg, _ = _load_full_skill_from(registry, s["name"])
        if isinstance(cfg, dict):
            lines.append(json.dumps({
                "name": cfg.get("name"),
                "id": cfg.get("id"),
                "type": cfg.get("type"),
                "description": cfg.get("description", ""),
                "parameters": cfg.get("parameters", {}),
                "workflow_steps": cfg.get("workflow_steps"),
            }, ensure_ascii=False))
    return "\n".join(lines)


def build_base_work_agent(
    *,
    request_key: str,          # "animal_request" | "plant_request"
    result_key: str,           # "animal_result" | "plant_result"
    domain: str,               # "animal" | "plant"
    domain_label: str,         # "陆生动物" | "陆生植物"
    state_cls,
    registry: List[Dict],
    planner_llm,
):
    """构建一个领域 base work 子图。

    内部执行字段命名与另一子图完全一致（命名空间可以相同），状态分离；
    仅通过 request_key / result_key 两个桥接通道与父图交换信息。
    """

    def _load(skill_name: str) -> tuple:
        return _load_full_skill_from(registry, skill_name)

    planner_system_prompt = SUB_PLANNER_SYSTEM_PROMPT.format(domain_label=domain_label)

    # ---------------- 规划器 ----------------

    def sub_planner_node(state):
        """任务规划器：根据用户输入 + 本子图技能目录，输出 JSON 决定如何执行。"""
        user_text = (state.get(request_key) or "").strip()
        catalog_text = _build_skill_catalog_text(registry) or "（本子图暂无技能）"

        # 重置内部执行状态（同一 checkpoint 命名空间下可能残留上一轮数据）
        update = {
            "domain": domain,
            "sub_delegated": True,
            "execution_mode": "idle",
            "selected_skill": None,
            "skill_config": None,
            "skill_params": {},
            "work_dir": "", "input_file": "", "history_file": "", "pa": "", "regional_level": "",
            "workflow_steps": [],
            "current_step_idx": 0,
            "step_outputs": {},
            "retry_count": {},
            "review_action": None,
            "review_feedback": None,
            "approved": None,
            "error": None,
            "final_output": None,
        }

        user_prompt = f"""技能目录（每行一个技能 JSON）：
{catalog_text}

用户输入（父图原样转发）：
{user_text}

请输出 JSON："""

        try:
            resp = planner_llm.invoke([
                SystemMessage(content=planner_system_prompt),
                HumanMessage(content=user_prompt),
            ])
            content = resp.content if hasattr(resp, 'content') else str(resp)
            plan = _parse_json_safely(content)
        except Exception as e:
            print(f"[DEBUG] planner({domain}) LLM 异常: {e}")
            plan = {}

        action = plan.get("action")
        print(f"[DEBUG] planner({domain}) action={action}, plan={json.dumps(plan, ensure_ascii=False)[:300]}")

        if action not in ("workflow", "single_skill"):
            reason = plan.get("reason") or "无法根据用户输入生成可执行规划，需要补充信息。"
            update["execution_mode"] = "need_info"
            update["final_output"] = {
                "skill": "任务规划器",
                "type": "need_info",
                "status": "need_info",
                "reason": reason,
            }
            return update

        mode = plan.get("mode") or ("single_skill" if action == "single_skill" else "workflow")
        skill_id = plan.get("skill_id") or plan.get("skill_name")
        params = plan.get("parameters") or {}
        task_name = plan.get("task_name")

        update["execution_mode"] = mode
        update["skill_params"] = params
        for k in ("work_dir", "input_file", "history_file", "pa", "regional_level"):
            if k in params and params[k]:
                update[k] = params[k]

        if mode == "single_skill":
            cfg, err = _load(skill_id)
            if cfg is None:
                update["execution_mode"] = "need_info"
                update["final_output"] = {"skill": skill_id, "type": "need_info", "status": "need_info",
                                          "reason": f"规划失败：{err}"}
                return update
            update["selected_skill"] = cfg.get("name")
            update["skill_config"] = cfg
            print(f"[DEBUG] planner({domain}) → single_skill '{cfg.get('name')}' params={params}")
            return update

        # workflow 模式
        steps = _normalize_workflow_steps(plan.get("workflow_steps") or [])
        if not steps:
            cfg, _ = _load(skill_id)
            if isinstance(cfg, dict) and cfg.get("workflow_steps"):
                steps = _normalize_workflow_steps(cfg["workflow_steps"])
                task_name = task_name or cfg.get("name")
        if not steps:
            update["execution_mode"] = "need_info"
            update["final_output"] = {"skill": skill_id, "type": "need_info", "status": "need_info",
                                      "reason": "规划失败：缺少 workflow_steps，无法执行。"}
            return update

        task_name = task_name or skill_id or f"{domain_label}数据分析"
        skill_config = {
            "name": task_name,
            "id": skill_id or "planned_workflow",
            "description": plan.get("reason") or "由子图规划器生成的工作流",
            "type": "workflow",
            "parameters": {
                "work_dir": "工作目录路径", "input_file": "输入文件名",
                "history_file": "历史资料文件名", "pa": "居留型起始标记",
                "regional_level": "省级保护级别"
            },
            "workflow_steps": steps,
        }
        update["selected_skill"] = task_name
        update["skill_config"] = skill_config
        update["workflow_steps"] = steps
        print(f"[DEBUG] planner({domain}) → workflow '{task_name}' 共 {len(steps)} 步")
        return update

    # ---------------- 执行层 ----------------

    def find_previous_skill_output(state, current_idx: int) -> Dict:
        steps = state["workflow_steps"]
        for i in range(current_idx - 1, -1, -1):
            if detect_step_type(steps[i]) != "review":
                step_key = f"step_{steps[i]['step']}"
                return state["step_outputs"].get(step_key, {})
        return {}

    def find_next_skill_steps(state, current_idx: int) -> List[Dict]:
        steps = state["workflow_steps"]
        next_steps = []
        for i in range(current_idx + 1, len(steps)):
            step = steps[i]
            if detect_step_type(step) != "review":
                cfg, _ = _load(step["use_skill"])
                if isinstance(cfg, dict):
                    next_steps.append({
                        "step_num": step["step"],
                        "skill_name": cfg.get("name", step["use_skill"]),
                        "skill_id": step["use_skill"],
                        "description": cfg.get("description", "")
                    })
        return next_steps

    def build_params(state, step: Dict, target_config: Dict) -> Dict:
        params = {
            "work_dir": state.get("work_dir", "."),
            "input_file": state.get("input_file", ""),
            "history_file": state.get("history_file", ""),
            "pa": state.get("pa", ""),
            "regional_level": state.get("regional_level", "")
        }
        work_dir = params["work_dir"]

        if "input_from" in step:
            input_from = step["input_from"]
            if isinstance(input_from, dict):
                for target_param, source in input_from.items():
                    if isinstance(source, str):
                        params[target_param] = source if os.path.isabs(source) else os.path.join(work_dir, source)
                    elif isinstance(source, dict) and "step" in source and "index" in source:
                        src_key = f"step_{source['step']}"
                        src_output = state["step_outputs"].get(src_key, {})
                        output_files = src_output.get("output_files", [])
                        if 0 <= source["index"] < len(output_files):
                            params[target_param] = output_files[source["index"]]
                        else:
                            print(f"Warning: step {source['step']} output_files[{source['index']}] not available")
            else:
                raw = step["input_from"]
                num_str = raw.replace("step", "") if isinstance(raw, str) else str(raw)
                try:
                    source_step_num = int(num_str)
                except Exception:
                    source_step_num = None
                if source_step_num is not None:
                    source_key = f"step_{source_step_num}"
                    source_output = state["step_outputs"].get(source_key, {})
                    output_files = source_output.get("output_files", [])
                    if output_files:
                        params["input_file"] = output_files[0]
                    else:
                        params["input_file"] = infer_input_file(state, source_step_num)

        for param_name in target_config.get("parameters", {}):
            if param_name not in params:
                params[param_name] = state.get(param_name, "")

        return params

    def infer_input_file(state, step_num: int) -> str:
        work_dir = state.get("work_dir", ".")
        file_map = {1: "列表.xlsx", 2: "名录.xlsx"}
        if domain == "plant":
            file_map = {1: "植物列表.xlsx", 2: "植物名录.xlsx"}
        else:
            file_map = {1: "动物列表.xlsx", 2: "动物名录.xlsx"}
        key = f"step_{step_num}"
        if key in state["step_outputs"]:
            files = state["step_outputs"][key].get("output_files", [])
            if files:
                return files[0]
        return os.path.join(work_dir, file_map.get(step_num, "input.xlsx"))

    def execute_skill(state, step: Dict, step_idx: int) -> Dict:
        target_skill_id = step["use_skill"]
        target_config, _json_path = _load(target_skill_id)

        if target_config is None:
            return {"error": f"无法加载技能 {target_skill_id}", "status": "error"}

        per_params = build_params(state, step, target_config)
        params = {k: v for k, v in per_params.items() if v != ""}
        print(f"[DEBUG] execute_skill({domain}) params: {params}")

        results = []
        for wf_step in target_config.get("workflow", []):
            tool_name = wf_step.get("tool")
            if tool_name == "run_python":
                script_args = substitute_params(wf_step["params"]["script_args"], params)
                script_args = _clean_unresolved_script_args(script_args)
                sa_str = " ".join(script_args) if isinstance(script_args, list) else script_args
                print(f"[DEBUG] execute_skill({domain}): run_python {wf_step['params']['script_path']} {sa_str}")
                result = run_python.invoke({"script_path": wf_step["params"]["script_path"], "script_args": sa_str})
                is_success = not result.startswith(("错误", "执行失败"))
                results.append({
                    "tool": "run_python",
                    "script": wf_step["params"]["script_path"],
                    "description": wf_step.get("description", ""),
                    "result": result,
                    "success": is_success
                })
                if not is_success:
                    return {
                        "error": f"步骤 {step['step']} 执行失败: {result}",
                        "status": "error",
                        "step_outputs": {
                            **state["step_outputs"],
                            f"step_{step['step']}": {
                                "step_idx": step_idx, "step_num": step["step"],
                                "skill_id": target_skill_id, "skill_name": target_config.get("name"),
                                "params": params, "results": results, "output_files": [],
                            }
                        }
                    }
            elif tool_name == "run_r":
                script_args = substitute_params(wf_step["params"]["script_args"], params)
                script_args = _clean_unresolved_script_args(script_args)
                sa_str = " ".join(script_args) if isinstance(script_args, list) else script_args
                print(f"[DEBUG] execute_skill({domain}): run_r {wf_step['params']['script_path']} {sa_str}")
                result = run_r.invoke({"script_path": wf_step["params"]["script_path"], "script_args": sa_str})
                is_success = not result.startswith(("错误", "执行失败"))
                results.append({
                    "tool": "run_r",
                    "script": wf_step["params"]["script_path"],
                    "description": wf_step.get("description", ""),
                    "result": result,
                    "success": is_success
                })
                if not is_success:
                    return {
                        "error": f"步骤 {step['step']} 执行失败: {result}",
                        "status": "error",
                        "step_outputs": {
                            **state["step_outputs"],
                            f"step_{step['step']}": {
                                "step_idx": step_idx, "step_num": step["step"],
                                "skill_id": target_skill_id, "skill_name": target_config.get("name"),
                                "params": params, "results": results, "output_files": [],
                            }
                        }
                    }

        output_files = extract_output_files(results, params)
        step_key = f"step_{step['step']}"
        print(f"[DEBUG] execute_skill({domain}) 输出: {output_files}")

        return {
            "step_outputs": {
                **state["step_outputs"],
                step_key: {
                    "step_idx": step_idx,
                    "step_num": step["step"],
                    "skill_id": target_skill_id,
                    "skill_name": target_config.get("name"),
                    "config_path": _json_path,
                    "params": params,
                    "results": results,
                    "output_files": output_files,
                }
            },
            "current_step_idx": step_idx + 1,
            "status": "step_completed"
        }

    def execute_review(state, step: Dict, step_idx: int) -> Dict:
        prev_output = find_previous_skill_output(state, step_idx)
        next_steps = find_next_skill_steps(state, step_idx)

        step_num = step['step']
        step_desc = step.get("description", "未命名步骤")
        workflow_name = state["skill_config"].get("name", "未命名工作流")
        retry_count = state["retry_count"].get(f"step_{prev_output.get('step_num', 'unknown')}", 0)

        lines = [
            "=" * 50,
            f"🔍 工作流审核请求 | {workflow_name}（{domain_label}）",
            "=" * 50,
            "",
            f"步骤编号: {step_num}",
            f"步骤描述: {step_desc}",
            f"重试次数: {retry_count}",
            "",
            "-" * 50,
            "📋 上一步执行结果:",
            "-" * 50,
        ]

        if prev_output:
            prev_step = prev_output.get('step_num', 'N/A')
            prev_status = prev_output.get('status', 'unknown')
            lines.append(f"  步骤: {prev_step}")
            lines.append(f"  状态: {prev_status}")
            output_content = prev_output.get('output', prev_output.get('result', {}))
            if isinstance(output_content, dict):
                for k, v in output_content.items():
                    v_str = str(v)[:500] + "..." if len(str(v)) > 500 else str(v)
                    lines.append(f"  {k}: {v_str}")
            else:
                content_str = str(output_content)[:1000]
                lines.append(f"  结果: {content_str}")
        else:
            lines.append("  （无上一步输出）")

        lines.extend(["", "-" * 50, "📎 后续待执行步骤:", "-" * 50])
        if next_steps:
            for i, ns in enumerate(next_steps, 1):
                lines.append(f"  {i}. 步骤 {ns.get('step_num', 'N/A')}: {ns.get('description', '未描述')}")
        else:
            lines.append("  （无后续步骤）")

        lines.extend([
            "",
            "=" * 50,
            "⚡ 可执行操作（请回复对应指令）:",
            "=" * 50,
            "  [通过 / continue / 确认]  → 确认结果正确，继续执行后续步骤",
            "  [重新执行 / retry / 重试]  → 重新执行上一步骤",
            "  [终止 / stop / 结束]      → 终止整个任务",
            "",
            "💬 附加反馈（可选）: 可在指令后补充说明原因或修改建议",
            "=" * 50,
        ])

        review_text = "\n".join(lines)

        response = interrupt({
            "review_type": "workflow_intermediate",
            "review_id": f"review_{step_num}",
            "domain": domain,
            "title": f"审核步骤 {step_num}: {step_desc}",
            "workflow_name": workflow_name,
            "content_text": review_text,
            "content_structured": {
                "previous_step": prev_output,
                "next_steps_preview": next_steps,
                "retry_count": retry_count
            }
        })

        action = parse_review_action(response)

        return {
            "approved": action == "continue",
            "review_feedback": response.get("feedback", ""),
            "review_action": action,
            "review_text": review_text,
            "step_outputs": {
                **state["step_outputs"],
                f"step_{step_num}": {
                    "type": "review",
                    "review_data": response,
                    "step_idx": step_idx
                }
            }
        }

    def execute_step(state) -> Dict:
        steps = state["workflow_steps"]
        current_idx = state["current_step_idx"]

        if current_idx >= len(steps):
            print(f"[DEBUG] execute_step({domain}): 所有步骤已完成")
            return {"status": "completed"}

        step = steps[current_idx]
        step_type = detect_step_type(step)
        print(f"[DEBUG] execute_step({domain}): 步骤 {current_idx + 1}/{len(steps)}, type={step_type}")

        if step_type == "review":
            return execute_review(state, step, current_idx)
        else:
            return execute_skill(state, step, current_idx)

    def step_executor_node(state):
        """步骤执行节点包装器，任务失败时触发中断进入人工审核。"""
        result = execute_step(state)
        status = result.get("status", "unknown")

        if status == "error":
            error_msg = result.get("error", "未知错误")
            step_num = state["workflow_steps"][state["current_step_idx"]]["step"]
            interrupt_request = {
                "type": "execution_error",
                "domain": domain,
                "step_num": step_num,
                "error": error_msg,
                "actions": {
                    "retry": "重新执行当前步骤（使用相同参数）",
                    "abort": "终止任务"
                }
            }
            # 暂停图，等待人工审核决定
            user_choice = interrupt(interrupt_request)
            action = user_choice.get("action")

            if action == "retry":
                step_key = f"step_{step_num}"
                cleaned_outputs = dict(state["step_outputs"])
                cleaned_outputs.pop(step_key, None)
                return {
                    "step_outputs": cleaned_outputs,
                    "current_step_idx": state["current_step_idx"],
                    "error": None,
                    "status": "retry_current",
                    "review_action": None,
                    "review_feedback": None
                }
            else:  # abort
                return {
                    "review_action": "abort",
                    "review_feedback": f"执行错误后人工终止: {error_msg}",
                    "final_output": {
                        "skill": state["selected_skill"],
                        "status": "aborted",
                        "reason": f"执行错误后人工终止: {error_msg}",
                        "step_outputs": state["step_outputs"]
                    }
                }

        if status == "step_completed":
            return {
                "current_step_idx": result["current_step_idx"],
                "step_outputs": result.get("step_outputs", {})
            }

        if status == "completed":
            return {
                "step_outputs": result.get("step_outputs", {}),
                "final_output": {
                    "skill": state["selected_skill"],
                    "status": "completed",
                    "step_outputs": state["step_outputs"]
                }
            }

        return result

    def single_executor_node(state):
        """单技能执行节点：执行 skill_config 中的 workflow 脚本。"""
        config = state["skill_config"]
        workflow = config.get("workflow", [])
        print(f"[DEBUG] single_executor({domain}): 开始执行, 共{len(workflow)}个步骤")
        results = []
        has_error = False

        skill_params = state.get("skill_params", {})
        global_params = {}
        for field in ("work_dir", "input_file", "pa", "history_file", "regional_level"):
            if field in state:
                global_params[field] = state[field]

        replace_vars = {}
        replace_vars.update(global_params)
        replace_vars.update(skill_params)
        if isinstance(state.get("skill_config"), dict):
            replace_vars.update(state["skill_config"].get("default_params", {}))

        for idx, step in enumerate(workflow):
            tool_name = step.get("tool")
            params = step.get("params", {})
            script_path = params.get("script_path", "未知")

            script_args = substitute_params(params.get("script_args", []), replace_vars)
            script_args = _clean_unresolved_script_args(script_args)
            sa_str = " ".join(script_args) if isinstance(script_args, list) else script_args

            print(f"[DEBUG] single_executor({domain}): 步骤{idx + 1} tool={tool_name} script={script_path} args={sa_str}")

            if tool_name == "run_python":
                result = run_python.invoke({"script_path": params["script_path"], "script_args": sa_str})
                is_success = not result.startswith(("错误", "执行失败"))
                results.append({"tool": "run_python", "script": params["script_path"],
                                "description": step.get("description", ""),
                                "result": result, "success": is_success})
                if not is_success:
                    has_error = True
                    break
            elif tool_name == "run_r":
                result = run_r.invoke({"script_path": params["script_path"], "script_args": sa_str})
                is_success = not result.startswith(("错误", "执行失败"))
                results.append({"tool": "run_r", "script": params["script_path"],
                                "description": step.get("description", ""),
                                "result": result, "success": is_success})
                if not is_success:
                    has_error = True
                    break

        if has_error:
            failed = results[-1]
            print(f"[DEBUG] single_executor({domain}): 执行失败 - {failed['result'][:200]}")
            return {
                "final_output": {
                    "skill": state["selected_skill"],
                    "type": "single",
                    "status": "error",
                    "error": f"脚本执行失败 [{failed['script']}]: {failed['result']}",
                    "results": results
                }
            }

        print(f"[DEBUG] single_executor({domain}): 执行成功, {len(results)}个步骤完成")
        return {
            "final_output": {
                "skill": state["selected_skill"],
                "type": "single",
                "status": "completed",
                "results": results
            }
        }

    def retry_node_fn(state, target_idx: int):
        steps = state["workflow_steps"]
        cleaned_outputs = dict(state["step_outputs"])
        for i in range(target_idx, len(steps)):
            step_key = f"step_{steps[i]['step']}"
            cleaned_outputs.pop(step_key, None)
        return {
            "current_step_idx": target_idx,
            "step_outputs": cleaned_outputs,
            "review_action": None,
            "review_feedback": None,
            "approved": None
        }

    def continue_node(state):
        return {
            "current_step_idx": state["current_step_idx"] + 1,
            "review_action": None,
            "review_feedback": None,
            "approved": None
        }

    def abort_node(state):
        reason = state.get("review_feedback") or "人工终止"
        existing = state.get("final_output")
        if isinstance(existing, dict) and existing.get("reason"):
            reason = existing["reason"]
        return {
            "final_output": {
                "skill": state["selected_skill"],
                "status": "aborted",
                "reason": reason,
                "completed_steps": state["step_outputs"]
            }
        }

    def complete_node(state):
        return {
            "final_output": {
                "skill": state["selected_skill"],
                "status": "completed",
                "step_outputs": state["step_outputs"]
            }
        }

    # ---------------- 收尾：结果写回父图桥接通道 ----------------

    def sub_finish_node(state):
        """任务结束：把结果写回 {result_key}（父图可见），并清空 {request_key}。"""
        request_text = (state.get(request_key) or "").strip()
        fo = state.get("final_output")
        if not isinstance(fo, dict):
            fo = {"skill": state.get("selected_skill"), "status": "unknown", "reason": "子图未产生有效结果"}

        payload = {
            "domain": domain,
            "request": request_text,
            "status": fo.get("status", "unknown"),
            "skill": fo.get("skill"),
            "final_output": fo,
        }
        print(f"[DEBUG] finish({domain}): 返回父图 result status={fo.get('status')}")
        return {
            result_key: payload,
            request_key: "",   # 清空任务文本，避免下一次委派误判
        }

    # ---------------- 路由 ----------------

    def entry_router(state):
        """入口路由：父图委派（request 非空）→ 规划器；否则按已有执行状态走。"""
        if state.get(request_key):
            return "planner"
        mode = state.get("execution_mode")
        if mode == "workflow" and state.get("workflow_steps"):
            return "executor"
        if mode == "single_skill" and state.get("skill_config"):
            return "single_executor"
        return "idle"

    def planner_router(state):
        mode = state.get("execution_mode")
        if mode == "workflow":
            return "executor"
        if mode == "single_skill":
            return "single_executor"
        return "finish"   # need_info / idle → 直接收尾返回结果

    def find_previous_skill_idx(state) -> Optional[int]:
        current_idx = state["current_step_idx"]
        steps = state["workflow_steps"]
        for i in range(current_idx - 1, -1, -1):
            if detect_step_type(steps[i]) == "skill":
                return i
        return None

    def workflow_router(state):
        if state["current_step_idx"] >= len(state["workflow_steps"]):
            return "workflow_complete"

        action = state.get("review_action")
        if action == "abort":
            return "workflow_abort"
        if action in ("retry", "retry_previous"):
            prev_idx = find_previous_skill_idx(state)
            if prev_idx is not None:
                return f"retry_step_{prev_idx}"
            return "workflow_abort"
        if action == "continue":
            return "continue_step"
        return "execute_step"

    # ---------------- 构建图 ----------------
    builder = StateGraph(state_cls)

    builder.add_node("planner", sub_planner_node)
    builder.add_node("executor", step_executor_node)
    builder.add_node("single_executor", single_executor_node)
    builder.add_node("continue", continue_node)
    builder.add_node("abort", abort_node)
    builder.add_node("complete", complete_node)
    builder.add_node("finish", sub_finish_node)

    for i in range(3):
        builder.add_node(f"retry_{i}", lambda s, idx=i: retry_node_fn(s, idx))

    builder.add_conditional_edges(START, entry_router, {
        "planner": "planner",
        "executor": "executor",
        "single_executor": "single_executor",
        "idle": END,
    })

    builder.add_conditional_edges("planner", planner_router, {
        "executor": "executor",
        "single_executor": "single_executor",
        "finish": "finish",
    })

    builder.add_conditional_edges("executor", workflow_router, {
        "workflow_complete": "complete",
        "workflow_abort": "abort",
        "execute_step": "executor",
        "continue_step": "continue",
        **{f"retry_step_{i}": f"retry_{i}" for i in range(3)}
    })

    builder.add_edge("continue", "executor")
    for i in range(3):
        builder.add_edge(f"retry_{i}", "executor")

    builder.add_edge("complete", "finish")
    builder.add_edge("abort", "finish")
    builder.add_edge("single_executor", "finish")
    builder.add_edge("finish", END)

    return builder.compile(checkpointer=checkpointer)


# ========== 编译两个子图 ==========
checkpointer = MemorySaver()

animal_base_work_agent = build_base_work_agent(
    request_key="animal_request",
    result_key="animal_result",
    domain="animal",
    domain_label="陆生动物",
    state_cls=AnimalAnalysisState,
    registry=animal_skill_registry,
    planner_llm=llm,
)

plant_base_work_agent = build_base_work_agent(
    request_key="plant_request",
    result_key="plant_result",
    domain="plant",
    domain_label="陆生植物",
    state_cls=PlantAnalysisState,
    registry=plant_skill_registry,
    planner_llm=llm,
)

print("[GRAPH] animal_base_work_agent / plant_base_work_agent 已编译")


# ============================================================
# 父图：混合检索 + 路由判断（不规划） + 结果汇报
# ============================================================

SOP_DOCX_PATH = os.getenv("SOP_DOCX_PATH", r"D:\PythonProject1\生态环境调查报告工作sop.docx")
CHROMA_DIR = os.getenv("CHROMA_DIR", str(Path(__file__).resolve().parent / "chroma_sop_db"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-v3")
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
EMBED_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")


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


_sop_chunks_cache = None
_semantic_store = None
_keyword_index = None


def _load_chunks() -> List[str]:
    global _sop_chunks_cache
    if _sop_chunks_cache is None:
        cache_file = Path(CHROMA_DIR) / "sop_chunks.json"
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                _sop_chunks_cache = json.load(f)
        else:
            try:
                import chromadb
                client = chromadb.PersistentClient(path=CHROMA_DIR)
                coll = client.get_collection("sop_semantic")
                data = coll.get(include=["documents"])
                _sop_chunks_cache = data.get("documents", []) or []
            except Exception as e:
                print(f"[VEC] 无法加载 chunks: {e}")
                _sop_chunks_cache = []
    return _sop_chunks_cache


def _get_retrievers():
    global _semantic_store, _keyword_index

    chunks = _load_chunks()

    if _keyword_index is None:
        _keyword_index = _BM25KeywordIndex(chunks)

    if _semantic_store is None:
        try:
            import chromadb
            client = chromadb.PersistentClient(path=CHROMA_DIR)

            _semantic_store = client.get_collection(
                name="sop_semantic",
                embedding_function=_DashScopeEmbeddingFunction(),
            )
            print(f"[VEC] 已加载已有语义索引（{len(chunks)} 块）")
        except Exception as e:
            print(f"[VEC] 加载语义索引失败，仅使用关键词检索: {e}")
            _semantic_store = None

    return _semantic_store, _keyword_index


def retrieve_sop_chunks(query: str, k: int = 5) -> List[str]:
    """混合检索：语义（DashScope text-embedding）+ 关键词（BM25），RRF 融合。"""
    chunks = _load_chunks()
    if not chunks:
        return []
    sem, kw = _get_retrievers()
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
            print(f"[VEC] 语义检索失败: {e}")

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


# ---------- 路由系统提示（只分类，不规划） ----------
PARENT_ROUTER_SYSTEM_PROMPT = """你是生态环境调查系统的"任务路由"，只负责把用户请求分发给对应处理方，【不做任何执行规划】。

根据【用户输入】与【混合检索命中的资料】输出 JSON：
{"route": "animal" | "plant" | "chat", "reason": "..."}

判定规则：
- "animal"：用户要求的是【动物类数据分析/报告撰写/名录处理任务】（涉及动物调查数据文件，如鸟类/兽类/两栖爬行监测样线、动物名录、居留型、动物多样性统计、动物报告撰写等，需要执行数据分析脚本完成任务）。
- "plant"：同理，植物类数据分析/报告撰写任务。
- "chat"：一般知识问答（例如询问某个物种的生活习性、生态学概念解释）、闲聊、查看文件、与数据分析脚本执行无关的请求。

注意：
1. 询问物种习性、科普知识等即使提到动物/植物名，也归入 "chat"（由父图根据检索资料直接回答）。
2. 只有"需要处理数据文件并执行分析/报告任务"的请求才路由给子图（animal/plant）。
3. 检索命中情况只能作为参考；即使数据库未命中，只要用户请求明显是动物/植物数据分析任务，仍按任务路由。

只输出 JSON，不要输出其他文字。"""


# ---------- 父图汇报/问答系统提示 ----------
PARENT_SYSTEM_PROMPT = """你是生态环境调查（陆生动物 / 陆生植物）项目的协调助手。不负责规划与执行：数据分析、报告撰写任务会原样转交给对应的领域子图（animal_base_work_agent / plant_base_work_agent）去规划并执行。你负责：

1. 一般问答：
   - 若混合检索命中了与问题相关的资料，请【基于检索资料回答】，并可简要说明资料要点；
   - 若检索未命中任何相关内容，而问题需要专业知识/资料支撑（如询问某物种的生活习性、特定方法细节、数据库收录内容等），请明确告知用户"当前数据库暂未收录相关内容"，不要编造答案；
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

def parent_retrieve_node(state: ParentState) -> Dict:
    """检索节点：混合检索 SOP 向量数据库（返回"无"或相关内容）。"""
    user_message = _get_last_human_text(state)
    docs = retrieve_sop_chunks(user_message, k=5)
    print(f"[DEBUG] parent_retrieve: 检索到 {len(docs)} 条资料" + ("" if docs else "（无）"))
    return {"retrieved_docs": docs}


def parent_router_node(state: ParentState) -> Dict:
    """路由节点：判断任务归属 animal / plant / chat（不规划执行细节）。"""
    user_text = _get_last_human_text(state)
    docs = state.get("retrieved_docs") or []
    doc_text = "\n\n".join(f"[资料{i + 1}] {d[:400]}" for i, d in enumerate(docs)) or "（无）"

    user_prompt = f"""检索到的 SOP 相关资料：
{doc_text}

用户输入：
{user_text}

请输出 JSON："""

    resp = parent_llm.invoke([
        SystemMessage(content=PARENT_ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ])
    content = resp.content if hasattr(resp, 'content') else str(resp)
    decision = _parse_json_safely(content)
    route = decision.get("route")
    if route not in ("animal", "plant"):
        route = "chat"
    print(f"[DEBUG] parent_router: route={route}, reason={decision.get('reason', '')}")
    return {"route": route, "route_reason": decision.get("reason", "")}


def parent_router_router(state: ParentState) -> str:
    """路由后走向：任务→分发；问答→直接回答。"""
    if state.get("route") in ("animal", "plant"):
        return "dispatch"
    return "chat"


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
    }
    # 注意：animal_result / plant_result 保留在父图状态中，供后续查看/汇报使用；
    # 下一次委派时 parent_dispatch_node 会先将其清空，因此不会串味。
    if msgs:
        update["messages"] = msgs
    return update


def parent_chat_node(state: ParentState):
    """父图聊天节点：回答一般问题 / 查看文件 / 汇报子图执行结果。"""
    docs = state.get("retrieved_docs") or []
    doc_block = ""
    if docs:
        doc_block = "\n\n检索到的相关资料：\n" + "\n\n".join(f"[资料{i + 1}] {d[:800]}" for i, d in enumerate(docs))
    else:
        doc_block = "\n\n检索到的相关资料：（无）"
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

# 检索 → 路由
parent_builder.add_edge(START, "parent_retrieve")
parent_builder.add_edge("parent_retrieve", "parent_router")

# 路由 → 分发 或 直接问答
parent_builder.add_conditional_edges("parent_router", parent_router_router, {
    "dispatch": "parent_dispatch",
    "chat": "parent_chat",
})

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

print("[GRAPH] 父图已编译：检索 → 路由 → (animal/plant 子图或直接问答) → 汇报")


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

            if interrupt_value.get("type") == "execution_error":
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
    print("父图：混合检索 + 路由（不规划）")
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
