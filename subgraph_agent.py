"""子图模块：陆生动物 / 陆生植物基础工作子图（base work agent）。

本文件由 agent3.5.py 拆分而来，便于独立维护与更新。

每个子图的执行流程：
    父图委派(request 桥接) → 任务规划器(输出 JSON)
      → executor / single_executor
      →（审核 / 执行失败 / 必填参数缺失 时通过 interrupt 询问用户）
      → finish 节点把结果写回 RESULT 桥接通道

父子图之间只通过 request_key / result_key 两个桥接通道交换最少信息，
子图内部执行字段与父图状态完全隔离。
"""

import os, json, subprocess, sys, re
from docx import Document
from pathlib import Path
from typing import TypedDict, Optional, Any, List, Dict, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage

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
# 必填参数检查（脚本参数缺失时先询问用户）
# ------------------------------------------------------------
# 技能的 parameters 字段形如 {参数名: 参数说明}，说明中含"必填"的即为必填项。
# 执行脚本前若必填项无值，则通过 interrupt 询问用户：
#   - 提供参数：用户逐项输入，未输入的项按"使用脚本默认参数"处理；
#   - 使用脚本默认参数：不传该参数，交由脚本自身默认值处理。
# 可选项缺失则直接忽略（沿用 _clean_unresolved_script_args 的行为）。
# ============================================================

REQUIRED_MARKERS = ("必填", "必选", "required")


def _param_is_required(meta: Any) -> bool:
    """判断单个参数的元信息是否标记为必填。"""
    if isinstance(meta, dict):
        if meta.get("required") is True:
            return True
        meta = meta.get("description", "")
    if isinstance(meta, str):
        low = meta.lower()
        return any(m in meta or m in low for m in REQUIRED_MARKERS)
    return False


def get_skill_param_meta(skill_config: Optional[Dict]) -> Dict[str, Any]:
    """返回 {参数名: 参数说明/元信息}。"""
    params = (skill_config or {}).get("parameters") or {}
    meta: Dict[str, Any] = {}
    if isinstance(params, dict):
        meta = dict(params)
    elif isinstance(params, list):
        for item in params:
            if isinstance(item, dict) and item.get("name"):
                meta[item["name"]] = item
    return meta


def get_required_params(skill_config: Optional[Dict]) -> List[str]:
    """从技能配置中解析出必填参数名列表。"""
    return [name for name, meta in get_skill_param_meta(skill_config).items()
            if _param_is_required(meta)]


def get_param_description(skill_config: Optional[Dict], name: str) -> str:
    """取参数说明文本（用于向用户展示）。"""
    meta = get_skill_param_meta(skill_config).get(name)
    if isinstance(meta, dict):
        return str(meta.get("description", "") or meta)
    return str(meta or "")


def find_missing_required_params(skill_config: Optional[Dict], params: Dict) -> List[str]:
    """返回 params 中值为空的必填参数名列表。"""
    return [name for name in get_required_params(skill_config)
            if not str((params or {}).get(name) or "").strip()]


def prompt_missing_required_params(domain: str, domain_label: str, skill_name: str,
                                   skill_config: Optional[Dict], params: Dict) -> Dict[str, Any]:
    """必填参数缺失时询问用户。

    Returns:
        用户补充的参数 {参数名: 值}；选择使用脚本默认参数时返回空 dict。
    """
    missing = find_missing_required_params(skill_config, params)
    if not missing:
        return {}

    response = interrupt({
        "type": "missing_required_params",
        "domain": domain,
        "domain_label": domain_label,
        "skill": skill_name,
        "message": "以下脚本必填参数缺失，请选择「提供参数」逐项输入，或选择「使用脚本默认参数」。",
        "missing_params": [
            {"name": name, "description": get_param_description(skill_config, name)}
            for name in missing
        ],
        "actions": {
            "provide": "逐项提供缺失的必填参数（留空的项将使用脚本默认参数）",
            "default": "全部使用脚本默认参数",
        },
    })

    if not isinstance(response, dict):
        return {}
    if response.get("action") == "default":
        print(f"[DEBUG] missing_params({domain}): 用户选择使用脚本默认参数 {missing}")
        return {}

    provided = response.get("params") or {}
    filled = {k: v for k, v in provided.items() if str(v or "").strip()}
    if filled:
        print(f"[DEBUG] missing_params({domain}): 用户提供参数 {list(filled.keys())}")
    still_default = [name for name in missing if name not in filled]
    if still_default:
        print(f"[DEBUG] missing_params({domain}): 其余项使用脚本默认参数 {still_default}")
    return filled


# ============================================================
# 子图构造器（动物 / 植物共用一套内部结构，命名空间一致）
# ------------------------------------------------------------
# 每个子图：
#   父图委派(REQUEST 桥接) → 任务规划器(输出 JSON) → executor / single_executor
#     →（审核 / 失败中断）→ finish 节点把结果写回 RESULT 桥接通道
# ============================================================

SUB_PLANNER_SYSTEM_PROMPT = """你是{domain_label}数据分析子图的"任务规划器"。
你只接收父图转发的【原始用户任务文本】。你的职责是：
根据用户输入 + 可用技能目录，匹配相关技能并【输出一个 JSON】决定如何执行。

【可用技能目录】（见用户消息，每行一个技能 JSON）。规划时必须遵守：
- use_skill / skill_id 只能使用技能目录中的 id 或 name。
- workflow_steps 每项形如：{{"step": 1, "type": "skill", "use_skill": "<id>", "input_from": <依赖的前序 step 编号>}}；
  审核步骤：{{"step": n, "type": "review", "description": "..."}}。没有依赖时可省略 input_from。
- 用户说"全工作流 / 完整流程 / 报告撰写"时，优先复用目录中 type=workflow 技能的 workflow_steps；
  否则按用户意图把多个 single 技能按顺序拼接成 workflow_steps。
- 用户只要求单一技能时，用 action=single_skill。
- parameters 只填用户明确给出的参数（work_dir / input_file / history_file / pa / regional_level 等），
  用户未提供的键一律省略（脚本有默认值，不要编造文件路径）。
  缺失的必填参数【不要在这里询问】，系统会在执行脚本前统一询问用户
  “提供参数还是使用脚本默认参数”。
- 无法确定要执行什么技能，或用户输入属于一般问答/闲聊（非数据分析任务）时，
  输出 {{"action": "chat", "reason": "需要向用户澄清/说明的内容"}}；
  仅缺少参数不构成 chat 的理由。

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
    checkpointer=None,
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
        # 必填参数缺失 → 先询问用户：提供参数 或 使用脚本默认参数
        per_params.update(prompt_missing_required_params(
            domain, domain_label,
            target_config.get("name") or target_skill_id,
            target_config, per_params,
        ))
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

        # 必填参数缺失 → 先询问用户：提供参数 或 使用脚本默认参数
        replace_vars.update(prompt_missing_required_params(
            domain, domain_label,
            state.get("selected_skill") or "未知技能",
            state.get("skill_config") or {}, replace_vars,
        ))

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
