import os, json, subprocess, re
import sys
from docx import Document
from pathlib import Path
from typing import TypedDict, Optional, Any, List, Dict, Annotated
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphInterrupt
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


# ========== 状态定义（补全所有字段） ==========
class AgentState(TypedDict):
    # 用户输入与技能选择
    messages: Annotated[list, add_messages]
    skill_registry: Optional[Dict]
    selected_skill: Optional[str]
    skill_config: Optional[Dict]

    # 工作目录与全局参数
    work_dir: str
    sample_file: str
    history_file: str
    pa: str
    regional_level: str

    # 工作流执行控制
    current_step_idx: int
    workflow: List[Dict]
    step_outputs: Dict[str, Any]

    # Review 状态
    review_feedback: Optional[str]
    approved: Optional[bool]
    review_action: Optional[str]

    # 重试追踪
    retry_count: Dict[str, int]  # 如 {"step_1": 0, "step_3": 1}
    retry_target_idx: Optional[int]  # 重试节点使用的目标索引

    final_output: Optional[Any]
    error: Optional[str]

#========== model ==========
llm = ChatOpenAI(
    model="qwen-plus",
    api_key= os.environ.get("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0
)

# ========== skills ==========
def discover_skills(skills_root: Path) -> List[Dict[str, Any]]:
    """扫描 skills/*.json，提取 name, description 和 json 路径"""
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

def _load_full_skill(skill_name: str) -> tuple:
    """加载技能的完整配置，返回 (skill_basic, full_skill_dict) 或 (None, error_msg)"""
    global skill_registry
    skill_basic = None
    for s in skill_registry:
        if s["name"] == skill_name or s.get("id") == skill_name:
            skill_basic = s
            break
    if not skill_basic:
        available = [s["name"] for s in skill_registry]
        return None, f"错误：未找到技能 '{skill_name}'，可用：{', '.join(available)}"

    json_path = Path(skill_basic["_json_path"])
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            full_skill = json.load(f)
    except Exception as e:
        return None, f"错误：无法加载技能配置 {json_path}: {e}"

    return full_skill, json_path


SKILLS_ROOT = Path("./skills")
skill_registry = discover_skills(SKILLS_ROOT)


# ========== system prompt ==========
def _build_system_prompt() -> str:
    prompt = (
        "你是一个文件操作与脚本执行助手。可用工具：list_files, read_xlsx, read_docx, run_python, run_r, run_skill, get_skill_info。\n\n"
        "当用户要求执行某个技能时，你必须严格按照以下流程操作：\n"
        "1. 先调用 get_skill_info 获取该技能的详细信息（参数列表、工作流等）。\n"
        "2. 结合用户的输入，判断哪些参数已经明确提供，哪些缺失。\n"
        "3. 如果有参数缺失，你不能直接调用skill，而必须生成一个提问，要求用户补充缺失参数，或者输入“使用默认值”以采用脚本默认值。此时停止所有工具调用，等待用户回复。\n"
        "4. 只有当所有参数都已明确（用户提供了值或明确说“使用默认值”）后，才能调用skill。调用时 skill_params 只包含用户明确提供的参数，不要包含用户未提供的参数。\n\n"
        "当前可用的技能（仅名称和功能描述）：\n"
    )
    global skill_registry
    if not skill_registry:
        prompt += "（暂无任何技能）\n"
    else:
        for skill in skill_registry:
            prompt += f"- {skill['name']}: {skill['description']}\n"
    prompt += "\n若用户未提及技能，只请求普通文件操作，则直接使用相应工具即可。"
    return prompt

SYSTEM_PROMPT = _build_system_prompt()

# ========== 工具函数（基于你提供的代码，已整合） ==========
@tool
def list_files(directory: str = ".") -> List[str]:
    """列出当前目录下的所有文件"""
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
def run_python(script_path: str, keywords: List[str]) -> str:
    """运行 python 脚本，传递关键词参数"""
    import subprocess, sys
    try:
        if not os.path.exists(script_path):
            return f"错误：脚本 {script_path} 不存在"
        cmd = [sys.executable, script_path] + keywords
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', timeout=60)
        if result.returncode != 0:
            return f"执行失败 (code {result.returncode}):\nSTDERR:\n{result.stderr}\nSTDOUT:\n{result.stdout}"
        return result.stdout
    except Exception as e:
        return f"执行异常：{e}"

@tool
def run_r(script_path: str, keywords: List[str]) -> str:
    """运行 R 脚本，传递关键词参数"""
    try:
        if not os.path.exists(script_path):
            return f"错误：脚本 {script_path} 不存在"
        rscript = "Rscript" if os.name != "nt" else "Rscript.exe"
        cmd = [rscript, script_path] + keywords
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            return f"执行失败 (code {result.returncode}):\nSTDERR:\n{result.stderr}\nSTDOUT:\n{result.stdout}"
        return result.stdout
    except FileNotFoundError:
        return "错误：未找到 Rscript，请确认R已安装并加入PATH"
    except subprocess.TimeoutExpired:
        return "错误：脚本执行超时（60秒）"
    except Exception as e:
        return f"执行异常：{e}"

@tool
def _get_skill_info(skill_name: str) -> str:
    """获取技能详细信息，包括参数和默认值说明"""
    global skill_registry
    full_skill, json_path = _load_full_skill(skill_name)
    if full_skill is None:
        return f"错误：技能 {skill_name}不存在"  # 错误消息
    info = f"技能名称: {full_skill.get('name', skill_name)}\n"
    info += f"描述: {full_skill.get('description', '')}\n"
    parameters = full_skill.get("parameters", {})
    if parameters:
        info += "参数:\n"
        for p, desc in parameters.items():
            info += f"  - {p}: {desc}\n"

    workflow_steps = full_skill.get("workflow")
    if workflow_steps:
        info += f"工作流步骤 (共{len(workflow_steps)}步):\n"
        for step in workflow_steps:
            info += f"  步骤{step['step']}: {step.get('use_skill', '未知')}"
            if step.get('review_point'):
                info += " [审核点]"
            info += "\n"
    elif full_skill.get("workflow"):
        info += f"工作流步骤 (共{len(full_skill['workflow'])}步):\n"
        for idx, step in enumerate(full_skill['workflow'], 1):
            info += f"  {idx}. {step.get('description', step.get('tool', '未知操作'))}\n"
    return info

@tool
def _run_skill(skill_name: str, skill_params: Dict[str, Any]) -> str:
    """运行 skill，传递关键词参数"""
    # 加载完整技能配置
    skill_basic, full_skill = _load_full_skill(skill_name)
    if full_skill is None:
        return skill_basic

    # 参数校验（只校验提供的参数名是否合法）
    param_defs = full_skill.get("parameters", {})
    valid_params = set(param_defs.keys()) if param_defs else set()
    if valid_params:
        invalid_keys = set(skill_params.keys()) - valid_params
        if invalid_keys:
            return (f"错误：技能 '{skill_name}' 不支持的参数: {', '.join(invalid_keys)}\n"
                    f"支持的参数: {', '.join(valid_params)}\n"
                    f"请使用正确的参数名重新调用 run_skill。")

    # 执行工作流或回退脚本
    workflow = full_skill.get("workflow")
    if workflow:
        outputs = []
        for step_idx, action in enumerate(workflow, 1):
            tool_name = action.get("tool")
            if not tool_name:
                outputs.append(f"步骤{step_idx} 缺少 tool 字段，跳过")
                continue

            # 应用参数替换，未提供的参数会返回 None 并被过滤
            raw_params = action.get("params", {})
            replaced_params = _replace_vars(raw_params, skill_params)
            # replaced_params 已经是过滤完 None 的字典
            desc = action.get("description", f"步骤{step_idx}: 执行 {tool_name}")
            print(f"\n[工作流] {desc}")
            print(f"[工作流] 工具: {tool_name}, 参数: {json.dumps(replaced_params, ensure_ascii=False)}")

            # 调用工具函数
            result = None
            if tool_name == "run_skill":
                result = "错误：工作流中不能嵌套调用 run_skill"
            elif tool_name == "get_skill_info":
                result = "错误：工作流中不能调用 get_skill_info"
            else:
                tool_func = TOOL_FUNCTIONS.get(tool_name)
                if not tool_func:
                    result = f"错误：未知工具 {tool_name}"
                else:
                    try:
                        # 对 run_python / run_r 的特殊处理：过滤空关键词
                        if tool_name in ("run_python", "run_r") and "keywords" in replaced_params:
                            # 只保留非空字符串
                            replaced_params["keywords"] = [kw for kw in replaced_params.get("keywords", []) if kw]
                        result = tool_func(**replaced_params)
                    except Exception as e:
                        result = f"执行异常：{e}"

            output_preview = result
            outputs.append(f"【步骤{step_idx}】{desc}\n结果: {output_preview}")

            if action.get("continue_on_error") is False and ("错误" in result or "失败" in result):
                outputs.append("工作流因错误终止。")
                break

        return "\n\n".join(outputs)
    else:
        # 向后兼容：使用 entry_script
        entry_script = full_skill.get("entry_script")
        if not entry_script:
            return "错误：技能未定义 workflow 或 entry_script"
        script_path = skill_basic["_json_path"].parent / entry_script
        if not script_path.exists():
            return f"错误：技能脚本不存在 {script_path}"

        # 只传递用户明确提供的参数
        cmd = [sys.executable, str(script_path)]
        for param_name, param_value in skill_params.items():
            cmd.extend([f"--{param_name}", str(param_value)])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', timeout=60)
            if result.returncode != 0:
                return f"执行失败 (code {result.returncode}):\nSTDERR:\n{result.stderr}\nSTDOUT:\n{result.stdout}"
            return result.stdout
        except subprocess.TimeoutExpired:
            return "错误：脚本执行超时（60秒）"
        except Exception as e:
            return f"执行异常：{e}"

TOOL_FUNCTIONS = {
    "list_files": list_files,
    "read_xlsx": read_xlsx,
    "read_docx": read_docx,
    "run_python": run_python,
    "run_r": run_r,
    "_run_skill":_run_skill
}

all_tools = [_get_skill_info, _run_skill, run_python, run_r, list_files, read_xlsx, read_docx]
tool_node = ToolNode(all_tools)

llm_with_tools = llm.bind_tools(all_tools)

# ========== 辅助函数 ==========
def substitute_params(keywords: List[str], params: Dict) -> List[str]:
    """将 keywords 中的 {var} 替换为 params 中的值"""
    result = []
    for kw in keywords:
        for key, val in params.items():
            kw = kw.replace(f"{{{key}}}", str(val))
        result.append(kw)
    return result

def _replace_vars(obj, params_vars: Dict[str, Any]):
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


def detect_step_type(step: Dict) -> str:
    """统一使用 type 字段标记步骤类型"""
    if step.get("type") == "review":
        return "review"
    return "skill"


def execute_step(state: AgentState) -> Dict:
    """
    通用步骤执行器：自动判断步骤类型并执行
    支持：skill / review / 任意扩展类型
    """
    steps = state["workflow"]
    current_idx = state["current_step_idx"]
    if current_idx >= len(steps):
        return {"status": "completed"}
    step = steps[current_idx]
    step_type = detect_step_type(step)
    if step_type == "review":
        return execute_review(state, step, current_idx)
    else:
        return execute_skill(state, step, current_idx)


def execute_skill(state: AgentState, step: Dict, step_idx: int) -> Dict:
    """执行 Skill 步骤"""
    target_skill_id = step["use_skill"]
    _, target_config = _load_full_skill(target_skill_id)
    if target_config is None:
        return {"error": f"无法加载技能 {target_skill_id}", "status": "error"}
    params = build_params(state, step, target_config)
    results = []
    for wf_step in target_config.get("workflow", []):
        if wf_step.get("tool") == "run_python":
            keywords = substitute_params(wf_step["params"]["keywords"], params)  # 需实现
            result = run_python(wf_step["params"]["script_path"], keywords)
            results.append({
                "tool": "run_python",
                "result": result,
                "success": not result.startswith(("错误", "执行失败"))
            })
    output_files = extract_output_files(results, params)
    step_key = f"step_{step['step']}"
    return {
        "step_outputs": {
            **state["step_outputs"],
            step_key: {
                "step_idx": step_idx,
                "step_num": step["step"],
                "skill_id": target_skill_id,
                "skill_name": target_config.get("name"),
                "params": params,
                "results": results,
                "output_files": output_files,
                "timestamp": "2026-05-15"
            }
        },
        "current_step_idx": step_idx + 1,
        "status": "step_completed"
    }


def single_skill_executor(state: AgentState):
    """执行单步 Skill（普通 workflow 类型）"""
    config = state["skill_config"]
    workflow = config.get("workflow", [])

    results = []
    for step in workflow:
        tool = step.get("tool")
        if tool == "run_python":
            params = step.get("params", {})
            # 替换模板变量
            keywords = substitute_params(params.get("keywords", []), state)
            result = run_python(params["script_path"], keywords)
            results.append({
                "tool": "run_python",
                "description": step.get("description", ""),
                "result": result
            })

    return {
        "final_output": {
            "skill": state["selected_skill"],
            "type": "single",
            "results": results
        }
    }


def execute_review(state: AgentState, step: Dict, step_idx: int) -> Dict:
    """
    执行 Review 步骤：中断等待人工
    关键：不推进 current_step_idx，由 resume 后路由决定
    """
    # 找到上一个非 review 步骤的输出
    prev_skill_output = find_previous_skill_output(state, step_idx)

    # 找到下一个要执行的步骤（用于展示）
    next_steps = find_next_skill_steps(state, step_idx)

    review_request = {
        "review_type": "workflow_intermediate",
        "review_id": f"review_{step['step']}",
        "title": step.get("description", f"步骤 {step['step']} 审核"),
        "workflow_name": state["skill_config"].get("name"),

        # 待审核内容
        "content": {
            "previous_step": prev_skill_output,
            "next_steps_preview": next_steps,
            "retry_count": state["retry_count"].get(f"step_{prev_skill_output.get('step_num')}", 0)
        },

        # 审核选项
        "actions": {
            "approve": {
                "label": "通过",
                "description": "确认结果正确，继续执行后续步骤",
                "next": "continue"
            },
            "retry": {
                "label": "重新执行",
                "description": "结果有误，重新执行上一步 Skill",
                "next": "retry_previous"
            },
            "abort": {
                "label": "终止工作流",
                "description": "终止整个工作流",
                "next": "abort"
            }
        }
    }

    # ===== 中断 =====
    review_response = interrupt(review_request)

    # 解析响应
    action = parse_review_action(review_response)

    return {
        "approved": action == "continue",
        "review_feedback": review_response.get("feedback", ""),
        "review_action": action,
        "step_outputs": {
            **state["step_outputs"],
            f"step_{step['step']}": {
                "type": "review",
                "review_data": review_response,
                "step_idx": step_idx
            }
        }
        # 不推进 current_step_idx！
    }


def find_previous_skill_output(state: AgentState, current_idx: int) -> Dict:
    """找到当前 review 步骤的上一个 skill 步骤输出"""
    steps = state["workflow_steps"]
    for i in range(current_idx - 1, -1, -1):
        if detect_step_type(steps[i]) != "review":
            step_key = f"step_{steps[i]['step']}"
            return state["step_outputs"].get(step_key, {})
    return {}


def find_next_skill_steps(state: AgentState, current_idx: int) -> List[Dict]:
    """找到 review 之后的所有步骤预览"""
    steps = state["workflow_steps"]
    next_steps = []
    for i in range(current_idx + 1, len(steps)):
        step = steps[i]
        if detect_step_type(step) != "review":
            _, cfg = _load_full_skill(step["use_skill"])
            if isinstance(cfg, dict):
                next_steps.append({
                    "step_num": step["step"],
                    "skill_name": cfg.get("name", step["use_skill"]),
                    "skill_id": step["use_skill"],
                    "description": cfg.get("description", "")
                })
    return next_steps


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


def build_params(state: AgentState, step: Dict, target_config: Dict) -> Dict:
    """构建执行参数，支持多步骤 input_from 链"""
    params = {
        "work_dir": state.get("work_dir", "."),
        "sample_file": state.get("sample_file", ""),
        "history_file": state.get("history_file", ""),
        "pa": state.get("pa", ""),
        "regional_level": state.get("regional_level", "")
    }
    # 处理 input_from
    if "input_from" in step:
        source_step_num = step["input_from"]
        source_key = f"step_{source_step_num}"
        source_output = state["step_outputs"].get(source_key, {})
        output_files = source_output.get("output_files", [])
        if output_files:
            params["input_file"] = output_files[0]
        else:
            # 回退：根据 step 推断
            params["input_file"] = infer_input_file(source_step_num, state)
    # 技能自定义参数
    for param_name in target_config.get("parameters", {}):
        if param_name not in params:
            params[param_name] = state.get(param_name, "")
    # 输出文件名推导
    if target_config.get("id") == "animal_simple_list":
        params["output_file"] = params.get("output_file", "动物列表.xlsx")
    elif target_config.get("id") == "animal_catalog_generate":
        params["output_file"] = params.get("output_file", "动物名录.xlsx")
    return params


def infer_input_file(step_num: int, state: AgentState) -> str:
    """从步骤历史推断输入文件"""
    work_dir = state.get("work_dir", ".")
    file_map = {1: "动物列表.xlsx", 3: "动物名录.xlsx"}
    key = f"step_{step_num}"
    if key in state["step_outputs"]:
        files = state["step_outputs"][key].get("output_files", [])
        if files:
            return files[0]
    return os.path.join(work_dir, file_map.get(step_num, "input.xlsx"))


def extract_output_files(results: List[Dict], params: Dict) -> List[str]:
    """提取输出文件路径"""
    files = []
    work_dir = params.get("work_dir", ".")
    if "output_file" in params:
        files.append(os.path.join(work_dir, params["output_file"]))
    for r in results:
        stdout = r.get("result", "")
        for line in stdout.split("\n"):
            for ext in [".xlsx", ".csv", ".txt", ".docx"]:
                if ext in line:
                    parts = line.split()
                    for part in parts:
                        if part.endswith(ext):
                            files.append(os.path.join(work_dir, part))
    return list(set(files))


# ========== 核心节点 =========
def step_executor_node(state: AgentState):
    """步骤执行节点包装器"""
    result = execute_step(state)

    if result.get("status") == "completed":
        return {
            "final_output": {
                "skill": state["selected_skill"],
                "status": "completed",
                "step_outputs": state["step_outputs"]
            }
        }

    if result.get("status") == "error":
        return {
            "error": result.get("error"),
            "final_output": {
                "status": "error",
                "message": result.get("error")
            }
        }

    # 正常步骤完成（skill 推进或 review 等待）
    return result


def retry_node(state: AgentState, target_idx: int):
    """重试节点：回退到指定步骤"""
    return {
        "current_step_idx": target_idx,
        "review_action": None,
        "review_feedback": None,
        "approved": None
    }


def continue_node(state: AgentState):
    """继续节点：从 review 后推进"""
    # 找到当前 review 步骤，推进到下一步
    current_idx = state["current_step_idx"]

    return {
        "current_step_idx": current_idx + 1,  # 跳过 review 步骤
        "review_action": None,
        "review_feedback": None,
        "approved": None
    }


def abort_node(state: AgentState):
    """终止节点"""
    return {
        "final_output": {
            "skill": state["selected_skill"],
            "status": "aborted",
            "reason": state.get("review_feedback", "用户终止"),
            "completed_steps": state["step_outputs"]
        }
    }


def complete_node(state: AgentState):
    """完成节点"""
    return {
        "final_output": {
            "skill": state["selected_skill"],
            "status": "completed",
            "step_outputs": state["step_outputs"]
        }
    }

# ========== 路由逻辑 ==========
def workflow_router(state: AgentState) -> str:
    """
    统一路由：处理所有情况（继续、重试、终止、完成）
    """
    # 检查是否完成
    if state["current_step_idx"] >= len(state["workflow"]):
        return "workflow_complete"

    # 处理 Review 结果
    action = state.get("review_action")

    if action == "abort":
        return "workflow_abort"

    if action == "retry_previous":
        # 找到上一个 skill 步骤的索引
        prev_idx = find_previous_skill_idx(state)
        if prev_idx is not None:
            # 增加重试计数
            prev_step = state["workflow"][prev_idx]
            step_key = f"step_{prev_step['step']}"
            retry_count = state.get("retry_count", {})
            retry_count[step_key] = retry_count.get(step_key, 0) + 1

            return f"retry_step_{prev_idx}"
        else:
            return "workflow_abort"  # 找不到上一步，终止

    if action == "continue":
        # 清除 review 状态，继续下一步
        return "continue_step"

    # 正常执行
    return "execute_step"


def find_previous_skill_idx(state: AgentState) -> Optional[int]:
    """找到上一个 skill 步骤的索引"""
    current_idx = state["current_step_idx"]
    steps = state["workflow"]

    for i in range(current_idx - 1, -1, -1):
        if detect_step_type(steps[i]) == "skill":
            return i

    return None

# ========== 构建图 ==========
builder = StateGraph(AgentState)

# 节点
builder.add_node("executor", step_executor_node)
builder.add_node("continue", continue_node)
builder.add_node("abort", abort_node)
builder.add_node("complete", complete_node)

# 动态注册重试节点
for i in range(3):
    builder.add_node(f"retry_{i}", lambda s, idx=i: retry_node(s, idx))

# 边
builder.add_edge(START, "executor")

builder.add_conditional_edges(
    "executor",
    workflow_router,
    {
        "workflow_complete": "complete",
        "workflow_abort": "abort",
        "execute_step": "executor",      # 继续执行下一步
        "continue_step": "continue",     # review 后继续
        **{f"retry_step_{i}": f"retry_{i}" for i in range(3)}
    }
)

builder.add_edge("continue", "executor")  # 继续后回到执行器
for i in range(3):
    builder.add_edge(f"retry_{i}", "executor")  # 重试后回到执行器

builder.add_edge("complete", END)
builder.add_edge("abort", END)


checkpointer = MemorySaver()
agent = builder.compile(checkpointer=checkpointer)


def _print_last_ai_message(agent, config):
    """输出最后一条 AI 消息"""
    state = agent.get_state(config)
    messages = state.values.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            print(f"\n🤖 助手: {msg.content}")
            return
    # 如果没有 AI 消息，检查 final_output
    final = state.values.get("final_output")
    if final:
        print(f"\n🤖 执行结果: {json.dumps(final, ensure_ascii=False, indent=2, default=str)}")


def _handle_interrupts(agent, config):
    """检查并处理图中的 interrupt 状态（人工审核）"""
    state = agent.get_state(config)
    for task in state.tasks:
        if not (hasattr(task, 'interrupts') and task.interrupts):
            continue

        for intr in task.interrupts:
            interrupt_value = intr.value if isinstance(intr, Interrupt) else intr
            print(f"\n📋 审核请求: {json.dumps(interrupt_value, ensure_ascii=False, indent=2, default=str)}")

            # 等待用户审核
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


# ========== 主入口 ==========
def run_interactive():
    """交互式运行"""
    config = {"configurable": {"thread_id": "interactive-1"}}

    print("=" * 60)
    print("野生动物调查数据分析助手")
    print("可用技能：", [s["name"] for s in skill_registry])
    print("输入 'quit' 退出")
    print("=" * 60)

    while True:
        user_input = input("\n🧑 你: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("再见！")
            break
        if not user_input:
            continue

        # 添加用户消息
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "execution_mode": "chat",
        }

        try:
            # stream 执行（interrupt 时图会自然暂停，不会抛异常）
            for _ in agent.stream(initial_state, config=config, stream_mode="values"):
                pass

            # 检查是否有 pending interrupt
            _handle_interrupts(agent, config)

            # 输出最后一条 AI 消息
            _print_last_ai_message(agent, config)

        except Exception as e:
            print(f"\n❌ 执行出错: {e}")


if __name__ == "__main__":
    run_interactive()