import os, json, subprocess, re, sys
from datetime import datetime
from docx import Document
from pathlib import Path
from typing import TypedDict, Optional, Any, List, Dict, Annotated
from dotenv import load_dotenv

import math

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt, Command, Interrupt
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage, HumanMessage
from langchain_deepseek import ChatDeepSeek


# ========== 状态定义 ==========
class AgentState(TypedDict):
    # 对话
    messages: Annotated[list, add_messages]

    # 模式: "chat" | "workflow" | "single_skill"
    execution_mode: str

    # 技能选择
    skill_registry: Optional[List[Dict]]
    selected_skill: Optional[str]
    skill_config: Optional[Dict]
    skill_params: Optional[Dict]

    # 工作目录与全局参数
    work_dir: str
    input_file: str
    history_file: str
    pa: str
    regional_level: str

    # 工作流执行控制
    current_step_idx: int
    workflow_steps: List[Dict]
    step_outputs: Dict[str, Any]

    # Review 状态
    review_feedback: Optional[str]
    approved: Optional[bool]
    review_action: Optional[str]

    # 重试追踪
    retry_count: Dict[str, int]
    retry_target_idx: Optional[int]

    # 结果
    final_output: Optional[Any]
    error: Optional[str]

    # 父图额外字段（子图会忽略）
    retrieved_docs: Optional[List[str]]
    plan: Optional[str]
    parent_plan: Optional[Dict]
    parent_action: Optional[str]
    parent_skill_registry: Optional[List[Dict]]
    subgraph_delegated: Optional[bool]


# ========== model ==========
load_dotenv(override=True)

llm = ChatDeepSeek(
    model="deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
    extra_body={"thinking": {"type": "disabled"}}
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
    """加载技能的完整配置"""
    global skill_registry
    skill_basic = None
    for s in skill_registry:
        if s["name"] == skill_name or s.get("id") == skill_name:
            skill_basic = s
            break

    if not skill_basic:
        available = [s["name"] for s in skill_registry]
        return None, f"错误：未找到技能 '{skill_name}'，可用：{', '.join(available)}"

    json_path = skill_basic["_json_path"]
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
    skill_list = ""
    if skill_registry:
        for skill in skill_registry:
            skill_list += f"  - {skill['name']}（id: {skill['id']}）: {skill['description']}\n"
    else:
        skill_list = "  （暂无技能）\n"

    prompt = f"""你是一个野生动物调查数据分析助手，帮助用户执行数据分析技能和文件操作。

    技能执行流程
    当用户要求执行某个技能时，你必须严格按以下步骤操作：
    1. 调用 get_skill_info 获取技能的详细信息（参数列表、工作流步骤等）
    2. 对比用户输入，判断哪些参数已提供、哪些缺失
    3. 如果有参数缺失，向用户提问，不要自行假设或编造
    4. 用户说"使用默认值"时，该参数可不传（脚本会使用内置默认值）
    5. 所有参数确认后，调用 launch_skill 启动执行

    严禁事项
    绝对不能编造、模拟或推测脚本的执行结果
    执行结果由系统消息告诉你，你不可以自己判断执行是否成功

    当前可用技能
    {skill_list} 

    如果用户的问题与技能无关，正常回答即可
    用户可能用简称或描述性语言指代技能，你需要匹配到正确的技能名称或id"""
    return prompt


SYSTEM_PROMPT = _build_system_prompt()


# ========== 工具定义 ==========
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


@tool
def get_skill_info(skill_name: str) -> str:
    """获取技能详细信息，包括参数说明和工作流步骤。在启动技能前必须先调用此工具了解参数要求。"""
    full_skill, json_path = _load_full_skill(skill_name)
    if full_skill is None:
        return json_path  # 此时存的是错误信息

    info = f"技能名称: {full_skill.get('name', skill_name)}\n"
    info += f"技能ID: {full_skill.get('id', '')}\n"
    info += f"描述: {full_skill.get('description', '无')}\n"
    info += f"类型: {full_skill.get('type', '未指定')}\n"
    info += f"配置文件: {json_path}\n"

    params = full_skill.get("parameters", {})
    if params:
        info += "\n参数:\n"
        for p, desc in params.items():
            info += f"  - {p}: {desc}\n"
    else:
        info += "\n参数: 无\n"

    workflow_steps = full_skill.get("workflow_steps")
    if workflow_steps:
        info += f"\n工作流步骤 (共{len(workflow_steps)}步):\n"
        for step in workflow_steps:
            step_type = "审核" if step.get("type") == "review" or step.get("review_point") else "执行"
            info += f"  步骤{step['step']} [{step_type}]: {step.get('description', step.get('use_skill', '未知'))}\n"
    elif full_skill.get("workflow"):
        wf = full_skill["workflow"]
        info += f"\n执行步骤 (共{len(wf)}步):\n"
        for idx, step in enumerate(wf, 1):
            info += f"  {idx}. {step.get('description', step.get('tool', '未知操作'))}\n"

    return info


@tool
def launch_skill(skill_name: str, params_json: str) -> str:
    """启动技能执行。当所有必要参数已确认后调用此工具。"""
    # 解析参数 JSON
    try:
        params = json.loads(params_json) if params_json.strip() else {}
    except json.JSONDecodeError as e:
        return f"参数格式错误，必须是合法JSON: {e}"

    # 验证技能是否存在
    full_skill, json_path = _load_full_skill(skill_name)
    if full_skill is None:
        return json_path  # 错误信息

    # 检查必填参数
    required_params = []
    for p, desc in full_skill.get("parameters", {}).items():
        if "必填" in desc or "必" in desc:
            required_params.append(p)

    missing = [p for p in required_params if p not in params]
    if missing:
        return f"缺少必填参数: {', '.join(missing)}。请先收集这些参数再启动。"

    # 用 _replace_vars 解析参数中的模板变量，解析失败的置空
    resolved_params = {}
    for key, val in params.items():
        if isinstance(val, str):
            resolved = _replace_vars(val, params)
            resolved_params[key] = resolved if resolved is not None else ""
        else:
            resolved_params[key] = val

    # 移除值为空字符串的参数（让脚本使用自身默认值）
    resolved_params = {k: v for k, v in resolved_params.items() if v != ""}

    return json.dumps({
        "action": "launch_skill",
        "skill_name": skill_name,
        "params": resolved_params,
        "skill_type": full_skill.get("type"),
        "has_workflow_steps": bool(full_skill.get("workflow_steps"))
    }, ensure_ascii=False)


# 工具列表和节点
all_tools = [get_skill_info, launch_skill, run_python, run_r, list_files, read_xlsx, read_docx]
tool_node = ToolNode(all_tools)
llm_with_tools = llm.bind_tools(all_tools)


# ========== 辅助函数 ==========

def detect_step_type(step: Dict) -> str:
    """统一检测步骤类型"""
    if step.get("use_skill") == "review":
        return "review"
    if step.get("review_point") is True and "use_skill" not in step:
        return "review"
    if step.get("type") == "review":
        return "review"
    return "skill"


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
            cfg, _ = _load_full_skill(step["use_skill"])
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
    """构建技能执行参数，支持从多个上游步骤按索引选择文件"""
    params = {
        "work_dir": state.get("work_dir", "."),
        "input_file": state.get("input_file", ""),
        "history_file": state.get("history_file", ""),
        "pa": state.get("pa", ""),
        "regional_level": state.get("regional_level", "")
    }

    work_dir = params["work_dir"]

    #print(f"步骤2开始前，{state.keys()} {state['step_outputs']}")

    # 处理 input_from
    if "input_from" in step:
        input_from = step["input_from"]

        # ----- 新格式：映射字典 -----
        if isinstance(input_from, dict):
            for target_param, source in input_from.items():
                # 情况1：直接指定文件名（字符串）
                if isinstance(source, str):
                    # 支持绝对路径或相对路径，若相对则拼接 work_dir
                    if os.path.isabs(source):
                        params[target_param] = source
                    else:
                        params[target_param] = os.path.join(work_dir, source)

                # 情况2：字典形式 {"step": s, "index": i}
                elif isinstance(source, dict) and "step" in source and "index" in source:
                    src_step = source["step"]
                    src_idx = source["index"]
                    src_key = f"step_{src_step}"
                    src_output = state["step_outputs"].get(src_key, {})
                    output_files = src_output.get("output_files", [])
                    if 0 <= src_idx < len(output_files):
                        params[target_param] = output_files[src_idx]
                    else:
                        print(f"Warning: step {src_step} output_files[{src_idx}] not available for {target_param}")
        else:
            # 将输入转为整数步骤号
            if "input_from" in step:
                raw = step["input_from"]
                if isinstance(raw, str):
                    num_str = raw.replace("step", "")
                    source_step_num = int(num_str)
                else:
                    source_step_num = int(raw)

                source_key = f"step_{source_step_num}"
                source_output = state["step_outputs"].get(source_key, {})
                output_files = source_output.get("output_files", [])
                if output_files:
                    params["input_file"] = output_files[0]
                else:
                    params["input_file"] = infer_input_file(source_step_num, state)

    print(f"新输入文件为{params['input_file']}")

    # 技能自定义参数（若未通过映射设置，则从 state 获取）
    for param_name in target_config.get("parameters", {}):
        if param_name not in params:
            params[param_name] = state.get(param_name, "")

    return params


def infer_input_file(step_num: int, state: AgentState) -> str:
    """从步骤历史推断输入文件"""
    work_dir = state.get("work_dir", ".")
    file_map = {1: "动物列表.xlsx", 2: "动物名录.xlsx"}
    key = f"step_{step_num}"
    if key in state["step_outputs"]:
        files = state["step_outputs"][key].get("output_files", [])
        if files:
            return files[0]
    return os.path.join(work_dir, file_map.get(step_num, "input.xlsx"))


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
            # 这个值未被替换，同时移除它前面的 flag（如 --ref_file）
            if cleaned and cleaned[-1].startswith('--'):
                cleaned.pop()
            continue
        cleaned.append(sa)
    return cleaned


# ========== LLM Agent 层 ==========

def _find_launch_tool_call(messages: list) -> Optional[Dict]:
    """从消息历史中找到最近的 launch_skill 工具调用"""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls'):
            for tc in msg.tool_calls:
                if tc["name"] == "launch_skill":
                    return tc
    return None


def _find_launch_tool_result(messages: list) -> Optional[str]:
    """从消息历史中找到 launch_skill 的 ToolMessage 内容"""
    # 找到 launch_skill 的 tool_call_id
    launch_tc = _find_launch_tool_call(messages)
    if not launch_tc:
        return None
    tc_id = launch_tc["id"]
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage) and msg.tool_call_id == tc_id:
            return msg.content
    return None


def _log_messages(state: AgentState, node_name: str = ""):
    """打印当前消息流的摘要，便于调试"""
    messages = state.get("messages", [])
    if not messages:
        return
    prefix = f"[MSG {node_name}]" if node_name else "[MSG]"
    for i, msg in enumerate(messages):
        role = type(msg).__name__
        if isinstance(msg, HumanMessage):
            role = "👤 Human"
        elif isinstance(msg, AIMessage):
            tc_info = ""
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                tc_names = [tc["name"] for tc in msg.tool_calls]
                tc_info = f" [调用工具: {', '.join(tc_names)}]"
            role = f"🤖 AI{tc_info}"
        elif isinstance(msg, ToolMessage):
            role = "🔧 Tool"
        elif isinstance(msg, SystemMessage):
            role = "📋 System"

        content = msg.content or ""
        # 截断长内容
        if len(content) > 150:
            content = content[:150] + "..."
        print(f"  {prefix} [{i}] {role}: {content}")


def chat_node(state: AgentState):
    """LLM Agent 节点：理解意图、调用工具、收集参数、决定行动"""
    # 构建消息列表
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)

    # 打印 LLM 响应
    if hasattr(response, 'tool_calls') and response.tool_calls:
        tc_summary = ", ".join(
            f"{tc['name']}({json.dumps(tc['args'], ensure_ascii=False)[:100]})" for tc in response.tool_calls)
        print(f"[DEBUG] chat_node: LLM 调用工具 → {tc_summary}")
    elif response.content:
        preview = response.content[:200]
        print(f"[DEBUG] chat_node: LLM 回复 → {preview}")

    return {"messages": [response]}


def chat_router(state: AgentState) -> str:
    """决定 chat 节点后的路由"""
    last_message = state["messages"][-1] if state["messages"] else None

    if not isinstance(last_message, AIMessage):
        return "respond"

    # LLM 做了工具调用
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        tool_names = [tc["name"] for tc in last_message.tool_calls]
        # 检查是否有 launch_skill 调用
        for tc in last_message.tool_calls:
            if tc["name"] == "launch_skill":
                print(f"[DEBUG] chat_router: 检测到 launch_skill 调用, 参数: {tc['args']}")
                return "tools_then_launch"  # 先让 ToolNode 执行，再 prepare
        print(f"[DEBUG] chat_router: 普通工具调用: {tool_names}")
        return "tools"

    # LLM 直接回复（无工具调用）
    content_preview = last_message.content[:100] if last_message.content else "(空)"

    # 拦截：LLM 未调用 launch_skill 却声称执行完成
    if last_message.content and any(
            keyword in last_message.content
            for keyword in ["执行完成", "已执行", "已完成", "运行完成", "处理完成"]
    ):
        if not state.get("final_output"):
            print(f"[DEBUG] chat_router: ⚠️ LLM 编造执行结果，插入纠正消息重新调用")
            # 不直接输出虚假回复，而是给 LLM 一个纠正指令让它重新调用 launch_skill
            # 这里返回 "retry" 路由到一个纠正节点
            return "retry_with_correction"

    print(f"[DEBUG] chat_router: LLM 直接回复, 内容: {content_preview}")
    return "respond"


def after_tools_router(state: AgentState) -> str:
    """工具执行后的路由：检查 launch_skill 的返回结果"""
    launch_tc = _find_launch_tool_call(state["messages"])
    if launch_tc:
        tc_id = launch_tc["id"]
        for msg in reversed(state["messages"]):
            if isinstance(msg, ToolMessage) and msg.tool_call_id == tc_id:
                content_preview = msg.content[:200] if msg.content else "(空)"
                print(f"[DEBUG] after_tools_router: launch_skill 返回: {content_preview}")
                if "launch_skill" in msg.content:
                    print("[DEBUG] after_tools_router → prepare_launch")
                    return "prepare_launch"
                # 返回的是错误信息（缺参数等）→ 回到 chat 让 LLM 处理
                print("[DEBUG] after_tools_router → chat (launch_skill 返回错误)")
                return "chat"

    print("[DEBUG] after_tools_router → chat (无 launch_skill)")
    return "chat"


def prepare_launch_node(state: AgentState):
    """处理 launch_skill 请求，设置工作流执行状态"""
    # 找到 launch_skill 的工具调用
    launch_tc = _find_launch_tool_call(state["messages"])

    if not launch_tc:
        print("[DEBUG] prepare_launch: 未找到 launch_skill 工具调用！")
        return {
            "execution_mode": "chat",
            "messages": [SystemMessage(content="未找到有效的技能启动请求，请重新操作。")]
        }

    skill_name = launch_tc["args"].get("skill_name", "")
    params_raw = launch_tc["args"].get("params_json", "{}")
    print(f"[DEBUG] prepare_launch: skill_name={skill_name}, params_raw={params_raw}")

    # 解析参数（LLM 可能传字符串或 dict）
    if isinstance(params_raw, str):
        try:
            params = json.loads(params_raw)
        except json.JSONDecodeError:
            params = {}
    elif isinstance(params_raw, dict):
        params = params_raw
    else:
        params = {}

        # 加载技能配置
    full_skill, json_path = _load_full_skill(skill_name)
    if full_skill is None:
        return {
            "execution_mode": "chat",
            "messages": [SystemMessage(content=f"无法加载技能 '{skill_name}': {json_path}")]
        }

    # 判断执行模式
    if full_skill.get("type") == "workflow" and full_skill.get("workflow_steps"):
        mode = "workflow"
        workflow_steps = full_skill["workflow_steps"]
    else:
        mode = "single_skill"
        workflow_steps = []

    print(f"[DEBUG] prepare_launch: mode={mode}, workflow_steps={len(workflow_steps)}步, params={params}")

    # 构建状态更新
    state_update = {
        "execution_mode": mode,
        "selected_skill": skill_name,
        "skill_config": full_skill,
        "workflow_steps": workflow_steps,
        "current_step_idx": 0,
        "step_outputs": {},
        "retry_count": {},
        "review_feedback": None,
        "approved": None,
        "review_action": None,
        "error": None,
        "final_output": None,
        "skill_params": params.copy(),
    }

    # 合并参数到状态
    param_fields = [
        "work_dir", "input_file", "history_file", "pa", "regional_level"
    ]
    for field in param_fields:
        if field in params and params[field]:
            state_update[field] = params[field]

    return state_update


def mode_router(state: AgentState) -> str:
    """根据 execution_mode 路由到对应的执行器"""
    mode = state.get("execution_mode", "chat")
    if mode == "workflow":
        return "workflow"
    elif mode == "single_skill":
        return "single_skill"
    else:
        return "chat"


# ========== 工作流执行层 ==========

def execute_step(state: AgentState) -> Dict:
    """通用步骤执行器"""
    steps = state["workflow_steps"]
    current_idx = state["current_step_idx"]

    if current_idx >= len(steps):
        print(f"[DEBUG] execute_step: 所有步骤已完成")
        return {"status": "completed"}

    step = steps[current_idx]
    step_type = detect_step_type(step)
    print(f"[DEBUG] execute_step: 步骤 {current_idx + 1}/{len(steps)}, type={step_type}, detail={step}")

    if step_type == "review":
        return execute_review(state, step, current_idx)
    else:
        return execute_skill(state, step, current_idx)


def execute_skill(state: AgentState, step: Dict, step_idx: int) -> Dict:
    """执行 Skill 步骤"""
    target_skill_id = step["use_skill"]
    target_config, _json_path = _load_full_skill(target_skill_id)

    if target_config is None:
        return {"error": f"无法加载技能 {target_skill_id}", "status": "error"}

    per_params = build_params(state, step, target_config)
    params = {k: v for k, v in per_params.items() if v != ""}
    print(f"execute_skill params :{params}")

    results = []
    for wf_step in target_config.get("workflow", []):
        tool_name = wf_step.get("tool")
        if tool_name == "run_python":
            script_args = substitute_params(wf_step["params"]["script_args"], params)
            script_args = _clean_unresolved_script_args(script_args)
            print(f"execute_skill script_args :{script_args}")
            sa_str = " ".join(script_args) if isinstance(script_args, list) else script_args
            print(f"[DEBUG] execute_skill: run_python {wf_step['params']['script_path']} script_args={sa_str}")
            result = run_python.invoke({"script_path": wf_step["params"]["script_path"], "script_args": sa_str})
            is_success = not result.startswith(("错误", "执行失败"))
            results.append({
                "tool": "run_python",
                "script": wf_step["params"]["script_path"],
                "description": wf_step.get("description", ""),
                "result": result,
                "success": is_success
            })
            # 执行失败，停止后续步骤
            if not is_success:
                return {
                    "error": f"步骤 {step['step']} 执行失败: {result}",
                    "status": "error",
                    "step_outputs": {
                        **state["step_outputs"],
                        f"step_{step['step']}": {
                            "step_idx": step_idx,
                            "step_num": step["step"],
                            "skill_id": target_skill_id,
                            "skill_name": target_config.get("name"),
                            "params": params,
                            "results": results,
                            "output_files": [],
                        }
                    }
                }

        elif tool_name == "run_r":
            script_args = substitute_params(wf_step["params"]["script_args"], params)
            script_args = _clean_unresolved_script_args(script_args)
            sa_str = " ".join(script_args) if isinstance(script_args, list) else script_args
            print(f"[DEBUG] execute_skill: run_r {wf_step['params']['script_path']} args={sa_str}")
            result = run_r.invoke({"script_path": wf_step["params"]["script_path"], "script_args": sa_str})
            is_success = not result.startswith(("错误", "执行失败"))
            results.append({
                "tool": "run_r",
                "script": wf_step["params"]["script_path"],
                "description": wf_step.get("description", ""),
                "result": result,
                "success": is_success
            })
            # 执行失败，停止后续步骤
            if not is_success:
                return {
                    "error": f"步骤 {step['step']} 执行失败: {result}",
                    "status": "error",
                    "step_outputs": {
                        **state["step_outputs"],
                        f"step_{step['step']}": {
                            "step_idx": step_idx,
                            "step_num": step["step"],
                            "skill_id": target_skill_id,
                            "skill_name": target_config.get("name"),
                            "params": params,
                            "results": results,
                            "output_files": [],
                        }
                    }
                }

    output_files = extract_output_files(results, params)
    step_key = f"step_{step['step']}"
    print(f"输出{output_files}")

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


def execute_review(state: AgentState, step: Dict, step_idx: int) -> Dict:
    """执行 Review 步骤：中断等待人工审核"""
    prev_output = find_previous_skill_output(state, step_idx)
    next_steps = find_next_skill_steps(state, step_idx)

    step_num = step['step']
    step_desc = step.get("description", "未命名步骤")
    workflow_name = state["skill_config"].get("name", "未命名工作流")
    retry_count = state["retry_count"].get(f"step_{prev_output.get('step_num', 'unknown')}", 0)

    lines = [
        "=" * 50,
        f"🔍 工作流审核请求 | {workflow_name}",
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

    # 添加上一步输出内容
    if prev_output:
        prev_step = prev_output.get('step_num', 'N/A')
        prev_status = prev_output.get('status', 'unknown')
        lines.append(f"  步骤: {prev_step}")
        lines.append(f"  状态: {prev_status}")

        # 添加输出内容（如果是字符串直接展示，如果是字典则格式化）
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

    lines.extend([
        "",
        "-" * 50,
        "📎 后续待执行步骤:",
        "-" * 50,
    ])

    if next_steps:
        for i, ns in enumerate(next_steps, 1):
            ns_step = ns.get('step', 'N/A')
            ns_desc = ns.get('description', '未描述')
            ns_type = ns.get('type', 'unknown')
            lines.append(f"  {i}. [{ns_type}] 步骤 {ns_step}: {ns_desc}")
    else:
        lines.append("  （无后续步骤）")

    lines.extend([
        "",
        "=" * 50,
        "⚡ 可执行操作（请回复对应指令）:",
        "=" * 50,
        "  [通过 / continue / 确认]  → 确认结果正确，继续执行后续步骤",
        "  [重新执行 / retry / 重试]  → 重新执行上一步骤",
        "  [终止 / stop / 结束]      → 终止整个工作流",
        "",
        "💬 附加反馈（可选）: 可在指令后补充说明原因或修改建议",
        "=" * 50,
    ])

    review_text = "\n".join(lines)

    # 发送文本形式的中断请求
    response = interrupt({
        "review_type": "workflow_intermediate",
        "review_id": f"review_{step_num}",
        "title": f"审核步骤 {step_num}: {step_desc}",
        "workflow_name": workflow_name,
        "content_text": review_text,  # 文本格式便于阅读
        "content_structured": {  # 保留结构化数据供程序解析
            "previous_step": prev_output,
            "next_steps_preview": next_steps,
            "retry_count": retry_count
        }
    })

    action = parse_review_action(response)

    # 构建返回结果
    return {
        "approved": action == "continue",
        "review_feedback": response.get("feedback", ""),
        "review_action": action,
        "review_text": review_text,  # 保留文本便于日志记录
        "step_outputs": {
            **state["step_outputs"],
            f"step_{step_num}": {
                "type": "review",
                "review_data": response,
                "step_idx": step_idx
            }
        }
    }


def step_executor_node(state: AgentState):
    """步骤执行节点包装器，支持错误中断与重试"""
    result = execute_step(state)
    status = result.get("status", "unknown")

    # 处理执行错误：中断等待用户决策
    if status == "error":
        error_msg = result.get("error", "未知错误")
        step_num = state["workflow_steps"][state["current_step_idx"]]["step"]
        interrupt_request = {
            "type": "execution_error",
            "step_num": step_num,
            "error": error_msg,
            "actions": {
                "retry": "重新执行当前步骤（使用相同参数）",
                "abort": "终止工作流"
            }
        }
        # 暂停图，等待用户输入
        user_choice = interrupt(interrupt_request)
        action = user_choice.get("action")

        if action == "retry":
            # 清除当前步骤的输出，保持索引不变
            step_key = f"step_{step_num}"
            cleaned_outputs = dict(state["step_outputs"])
            cleaned_outputs.pop(step_key, None)
            return {
                "step_outputs": cleaned_outputs,
                "current_step_idx": state["current_step_idx"],
                "error": None,
                "status": "retry_current",  # 触发重试
                "review_action": None,
                "review_feedback": None
            }
        else:  # abort
            return {
                "final_output": {
                    "skill": state["selected_skill"],
                    "status": "aborted",
                    "reason": f"执行错误后用户终止: {error_msg}",
                    "step_outputs": state["step_outputs"]
                }
            }

    # 正常步骤完成推进索引
    if status == "step_completed":
        return {"current_step_idx": result["current_step_idx"],
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


def single_executor_node(state: AgentState):
    """单个步骤技能执行节点，支持任意用户参数"""
    config = state["skill_config"]
    workflow = config.get("workflow", [])
    print(f"[DEBUG] single_executor: 开始执行, 共{len(workflow)}个步骤")
    results = []
    has_error = False

    # 获取用户提供的所有参数（优先级最高）
    skill_params = state.get("skill_params", {})

    # 同时保留从 state 顶层获取的全局参数（向后兼容）
    global_params = {}
    global_fields = ["work_dir", "input_file", "pa", "history_file", "regional_level"]
    for field in global_fields:
        if field in state:
            global_params[field] = state[field]

    # 合并：用户参数 > 全局参数 > 技能默认参数（如果有）
    replace_vars = {}
    replace_vars.update(global_params)
    replace_vars.update(skill_params)

    # 如果技能配置中有默认参数，也加入（但会被用户参数覆盖）
    if isinstance(state.get("skill_config"), dict):
        defaults = state["skill_config"].get("default_params", {})
        replace_vars.update(defaults)

    for idx, step in enumerate(workflow):
        tool_name = step.get("tool")
        params = step.get("params", {})
        script_path = params.get("script_path", "未知")

        # 替换 script_args 中的模板变量（例如 {input_file1}）
        script_args = substitute_params(params.get("script_args", []), replace_vars)
        script_args = _clean_unresolved_script_args(script_args)
        sa_str = " ".join(script_args) if isinstance(script_args, list) else script_args

        print(f"[DEBUG] single_executor: 步骤{idx + 1} tool={tool_name} script={script_path}")
        print(f"[DEBUG] single_executor: 命令行参数: {sa_str}")

        if tool_name == "run_python":
            result = run_python.invoke({"script_path": params["script_path"], "script_args": sa_str})
            is_success = not result.startswith(("错误", "执行失败"))
            results.append({
                "tool": "run_python",
                "script": params["script_path"],
                "description": step.get("description", ""),
                "result": result,
                "success": is_success
            })
            if not is_success:
                has_error = True
                break
        elif tool_name == "run_r":
            result = run_r.invoke({"script_path": params["script_path"], "script_args": sa_str})
            is_success = not result.startswith(("错误", "执行失败"))
            results.append({
                "tool": "run_r",
                "script": params["script_path"],
                "description": step.get("description", ""),
                "result": result,
                "success": is_success
            })
            if not is_success:
                has_error = True
                break

    if has_error:
        failed = results[-1]
        print(f"[DEBUG] single_executor: 执行失败 - {failed['result'][:200]}")
        return {
            "final_output": {
                "skill": state["selected_skill"],
                "type": "single",
                "status": "error",
                "error": f"脚本执行失败 [{failed['script']}]: {failed['result']}",
                "results": results
            }
        }

    print(f"[DEBUG] single_executor: 执行成功, {len(results)}个步骤完成")
    return {
        "final_output": {
            "skill": state["selected_skill"],
            "type": "single",
            "status": "completed",
            "results": results
        }
    }


def retry_node_fn(state: AgentState, target_idx: int):
    """重试节点：回退到指定步骤，清理之后的输出"""
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


def continue_node(state: AgentState):
    """继续节点：跳过 review 步骤"""
    current_idx = state["current_step_idx"]
    return {
        "current_step_idx": current_idx + 1,
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


def correction_node(state: AgentState):
    """纠正节点：当 LLM 编造执行结果时，插入纠正消息强制其调用 launch_skill"""
    correction_msg = SystemMessage(
        content="⚠️ 你刚才声称技能已执行，但你并没有调用 launch_skill 工具！"
                "脚本不会自动执行，你必须调用 launch_skill 工具才能真正运行脚本。"
                "请立即调用 launch_skill 工具来启动技能执行，不要只描述结果。"
    )
    return {"messages": [correction_msg]}


def report_results_node(state: AgentState):
    """执行结束节点：重置模式，结果由父图 plan 节点反馈给用户"""
    return {"execution_mode": "chat"}


# ========== 路由逻辑 ==========

def find_previous_skill_idx(state: AgentState) -> Optional[int]:
    """找到上一个 skill 步骤的索引"""
    current_idx = state["current_step_idx"]
    steps = state["workflow_steps"]
    for i in range(current_idx - 1, -1, -1):
        if detect_step_type(steps[i]) == "skill":
            return i
    return None


def workflow_router(state: AgentState) -> str:
    """工作流执行路由"""
    # 步骤全部完成
    if state["current_step_idx"] >= len(state["workflow_steps"]):
        return "workflow_complete"

    # 处理 Review 结果
    action = state.get("review_action")
    if action == "abort":
        return "workflow_abort"

    if action in ("retry", "retry_previous"):
        prev_idx = find_previous_skill_idx(state)
        if prev_idx is not None:
            return f"retry_step_{prev_idx}"
        else:
            return "workflow_abort"

    if action == "continue":
        return "continue_step"

    # 正常执行下一步
    return "execute_step"


# ============================================================
# 修改子图入口：增加条件路由，使父图可以直接分发工作流
# ============================================================

def subgraph_entry_router(state: AgentState) -> str:
    """
    根据状态决定子图从哪个节点开始。
    如果状态中已有 workflow_steps 且 execution_mode 为 workflow 或 single_skill，
    则直接进入执行器，跳过 chat 层。
    """
    mode = state.get("execution_mode", "chat")
    if mode == "workflow" and state.get("workflow_steps"):
        return "executor"
    elif mode == "single_skill" and state.get("skill_config"):
        return "single_executor"
    else:
        return "sub_animal_date_analysis_chat"


# ========== 构建子图 ==========
builder = StateGraph(AgentState)

# --- Chat 层节点 ---
builder.add_node("sub_animal_date_analysis_chat", chat_node)
builder.add_node("tools", tool_node)
builder.add_node("prepare_launch", prepare_launch_node)
builder.add_node("correction", correction_node)

# --- 执行层节点 ---
builder.add_node("executor", step_executor_node)
builder.add_node("single_executor", single_executor_node)
builder.add_node("continue", continue_node)
builder.add_node("abort", abort_node)
builder.add_node("complete", complete_node)
builder.add_node("report_results", report_results_node)

# 动态注册重试节点
for i in range(3):
    builder.add_node(f"retry_{i}", lambda s, idx=i: retry_node_fn(s, idx))

# --- 边 ---
# 入口 → 条件路由
builder.add_conditional_edges(START, subgraph_entry_router, {
    "sub_animal_date_analysis_chat": "sub_animal_date_analysis_chat",
    "executor": "executor",
    "single_executor": "single_executor",
})

# chat 路由
builder.add_conditional_edges("sub_animal_date_analysis_chat", chat_router, {
    "tools": "tools",
    "tools_then_launch": "tools",
    "prepare_launch": "prepare_launch",
    "retry_with_correction": "correction",
    "respond": END,
})

# 纠正节点 → 回到 chat
builder.add_edge("correction", "sub_animal_date_analysis_chat")

# tools 执行后路由
builder.add_conditional_edges("tools", after_tools_router, {
    "chat": "sub_animal_date_analysis_chat",
    "prepare_launch": "prepare_launch",
})

# prepare_launch → 根据 mode 路由
builder.add_conditional_edges("prepare_launch", mode_router, {
    "workflow": "executor",
    "single_skill": "single_executor",
    "chat": "sub_animal_date_analysis_chat",
})

# --- 工作流执行路由 ---
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

# 完成和终止 → 报告结果 → 回到 chat
builder.add_edge("complete", "report_results")
builder.add_edge("abort", "report_results")
builder.add_edge("single_executor", "report_results")

# 报告结果后结束子图，由父图 plan 节点反馈结果
builder.add_edge("report_results", END)

# 编译子图
checkpointer = MemorySaver()
animal_date_analysis_agent = builder.compile(checkpointer=checkpointer)


# ============================================================
# 父图：负责检索与规划，并将任务分发至子图
# ============================================================

# ============================================================
# 父图：独立的协调层，不复用子图的节点、工具或技能配置
# ============================================================

# ---------- 混合检索（语义 + 关键词，RRF 融合） ----------
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


def _read_sop_chunks() -> List[str]:
    """读取 SOP 文档并按段落分块；文件缺失时回退到技能目录描述。"""
    chunks = []
    if os.path.exists(SOP_DOCX_PATH):
        try:
            doc = Document(SOP_DOCX_PATH)
            buf = []
            for p in doc.paragraphs:
                t = p.text.strip()
                if not t:
                    continue
                buf.append(t)
                if sum(len(x) for x in buf) >= 400:
                    chunks.append("\n".join(buf))
                    buf = []
            if buf:
                chunks.append("\n".join(buf))
            for tbl in doc.tables:
                rows = [" | ".join(c.text.strip() for c in row.cells) for row in tbl.rows]
                if rows:
                    chunks.append("\n".join(rows))
            print(f"[VEC] 已从 SOP 文档加载 {len(chunks)} 个文本块: {SOP_DOCX_PATH}")
        except Exception as e:
            print(f"[VEC] 读取 SOP 文档失败 {SOP_DOCX_PATH}: {e}")

    if not chunks:
        for s in skill_registry:
            cfg, _ = _load_full_skill(s["name"])
            if isinstance(cfg, dict):
                chunks.append(json.dumps(cfg, ensure_ascii=False))
        print(f"[VEC] 未找到 SOP 文档，回退使用 {len(chunks)} 个技能描述作为知识库")
    return chunks


_sop_chunks_cache = None
_semantic_store = None
_keyword_index = None


def _load_chunks() -> List[str]:
    global _sop_chunks_cache
    if _sop_chunks_cache is None:
        _sop_chunks_cache = _read_sop_chunks()
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
            try:
                client.delete_collection("sop_semantic")
            except Exception:
                pass
            coll = client.create_collection("sop_semantic", embedding_function=_DashScopeEmbeddingFunction())
            if chunks:
                vectors = _DashScopeEmbeddingFunction()(chunks)
                coll.add(ids=[f"c{i}" for i in range(len(chunks))], documents=chunks, embeddings=vectors)
            _semantic_store = coll
            print(f"[VEC] 语义索引已构建（{len(chunks)} 块，DashScope {EMBED_MODEL}）")
        except Exception as e:
            print(f"[VEC] 语义索引构建失败，仅使用关键词检索: {e}")
            _semantic_store = None
    return _semantic_store, _keyword_index


def retrieve_sop_chunks(query: str, k: int = 4) -> List[str]:
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


def load_parent_skill_catalog() -> List[Dict]:
    """父图规划所需的完整技能目录（含名称/id/描述/参数/工作流步骤）。"""
    catalog = []
    for s in skill_registry:
        cfg, _ = _load_full_skill(s["name"])
        if isinstance(cfg, dict):
            catalog.append(cfg)
    return catalog


parent_skill_registry = load_parent_skill_catalog()


# ---------- 父图 LLM（独立实例，可使用不同模型） ----------

parent_llm = ChatDeepSeek(
    model="deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
    extra_body={"thinking": {"type": "disabled"}}
)


# ---------- 父图工具集（仅用于聊天/汇报/文件操作，与子图独立） ----------

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


# ---------- 父图 System Prompt（聊天/汇报用） ----------

def _build_parent_system_prompt() -> str:
    skill_list = ""
    for cfg in parent_skill_registry:
        skill_list += f"  - {cfg.get('name')}（id: {cfg.get('id')}）: {cfg.get('description', '')}\n"
    if not skill_list:
        skill_list = "  （暂无技能）\n"

    return f"""你是野生动物调查项目的协调助手。数据分析任务由规划器（planner）负责制定工作流并交给数据分析子图执行，你负责：

1. 回答用户的一般性问题（结合检索资料回答；资料不足时使用自身知识，但不要编造数据分析结果）
2. 用户要求查看目录/读取文件时，使用 parent_list_files / parent_read_file_content 工具
3. 数据分析子图执行完成后，根据系统给出的执行结果，向用户清晰、友好地汇报

当前系统可用技能：
{skill_list}

注意：不要自行模拟或编造数据分析结果；数据分析结果以系统消息为准。"""


PARENT_SYSTEM_PROMPT = _build_parent_system_prompt()


# ---------- 规划器（LLM：检索资料 + 技能目录 → 工作流计划） ----------

def _build_skill_catalog_text() -> str:
    lines = []
    for cfg in parent_skill_registry:
        lines.append(json.dumps({
            "name": cfg.get("name"),
            "id": cfg.get("id"),
            "type": cfg.get("type"),
            "description": cfg.get("description", ""),
            "parameters": cfg.get("parameters", {}),
            "workflow_steps": cfg.get("workflow_steps"),
        }, ensure_ascii=False))
    return "\n".join(lines)


PLANNER_SYSTEM_PROMPT = """你是野生动物调查数据分析系统的"规划器"。根据用户输入，输出一个 JSON 决定如何执行。

【可用的技能目录】（见用户消息）。规划时必须遵守：
- use_skill 只能使用技能目录中的 id 或 name。
- workflow_steps 每项形如：{"step": 1, "type": "skill", "use_skill": "<id>", "input_from": <依赖的上一步step编号>}；审核步骤：{"step": n, "type": "review", "description": "..."}。没有依赖时可省略 input_from。
- 参数 parameters 从用户输入提取；用户未提供的键省略。
- 用户说"全工作流/完整流程"时，复用目录中 type=workflow 的技能（如 full_animal_workflow）的 workflow_steps。
- 用户说"已有动物列表/已完成列表"时，跳过"动物列表合并"，从"动物名录生成"开始。
- 用户只要求单一技能（如"动物名录分析"）时，用 action=single_skill。
- 无法确定工作流且缺少必要参数时，用 action=chat 交给协调助手向用户澄清；一般性问题（概念解释、闲聊、查看文件等）也用 action=chat。

输出 JSON 三选一：
{"action": "workflow", "task_name": "...", "mode": "workflow", "skill_id": "...", "parameters": {...}, "workflow_steps": [...]}
或
{"action": "single_skill", "task_name": "...", "mode": "single_skill", "skill_id": "...", "parameters": {...}}
或
{"action": "chat"}

只输出 JSON，不要输出其他文字。"""


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


# ---------- 父图节点函数 ----------

def parent_retrieve_node(state: AgentState) -> Dict:
    """检索节点：从 SOP 向量数据库检索相关资料。"""
    user_message = ""
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            user_message = m.content
            break
    docs = retrieve_sop_chunks(user_message, k=4)
    print(f"[DEBUG] parent_retrieve: 检索到 {len(docs)} 条资料")
    return {"retrieved_docs": docs}


def parent_planner_node(state: AgentState) -> Dict:
    """规划节点：LLM 根据用户输入 + 检索资料 + 技能目录，生成工作流计划。"""
    user_text = ""
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            user_text = m.content
            break
    docs = state.get("retrieved_docs") or []
    doc_text = "\n\n".join(f"[资料{i + 1}] {d[:600]}" for i, d in enumerate(docs))

    user_prompt = f"""技能目录（每行一个技能 JSON）：
{_build_skill_catalog_text()}

检索到的 SOP 相关资料：
{doc_text or "（无）"}

用户输入：
{user_text}

请输出 JSON："""

    resp = parent_llm.invoke([
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ])
    content = resp.content if hasattr(resp, 'content') else str(resp)
    plan = _parse_json_safely(content)
    print(f"[DEBUG] parent_planner: 计划 action={plan.get('action')}, mode={plan.get('mode')}")

    if plan.get("action") in ("workflow", "single_skill"):
        return {"parent_plan": plan, "parent_action": plan["action"]}
    return {"parent_plan": {}, "parent_action": "chat"}


def parent_planner_router(state: AgentState) -> str:
    if state.get("parent_action") in ("workflow", "single_skill"):
        return "dispatch"
    return "chat"


def _normalize_workflow_steps(steps: List[Dict]) -> List[Dict]:
    """规整工作流步骤：移除 input_from 为 None/0/空字符串的依赖标记。"""
    clean = []
    for st in steps:
        s = dict(st)
        if s.get("input_from") in (None, 0, ""):
            s.pop("input_from", None)
        clean.append(s)
    return clean


def parent_dispatch_node(state: AgentState) -> Dict:
    """分发节点：把规划结果转换为子图可直接执行的执行状态。"""
    plan = state.get("parent_plan") or {}
    mode = plan.get("mode") or ("single_skill" if plan.get("action") == "single_skill" else "workflow")
    params = plan.get("parameters") or {}
    skill_id = plan.get("skill_id") or plan.get("skill_name")

    state_update = {
        "skill_params": params,
        "current_step_idx": 0,
        "step_outputs": {},
        "retry_count": {},
        "final_output": None,
        "error": None,
        "subgraph_delegated": True,
    }
    for k in ("work_dir", "input_file", "history_file", "pa", "regional_level"):
        if k in params and params[k]:
            state_update[k] = params[k]

    if mode == "single_skill":
        cfg, err = _load_full_skill(skill_id)
        if cfg is None:
            return {"messages": [SystemMessage(content=f"规划失败：{err}")], "parent_action": "chat", "subgraph_delegated": False}
        state_update.update({
            "execution_mode": "single_skill",
            "selected_skill": cfg.get("name"),
            "skill_config": cfg,
            "workflow_steps": [],
        })
        print(f"[DEBUG] parent_dispatch: 单技能 {cfg.get('name')}")
        return state_update

    # workflow 模式
    steps = _normalize_workflow_steps(plan.get("workflow_steps") or [])
    task_name = plan.get("task_name")
    if not steps:
        cfg, _ = _load_full_skill(skill_id)
        if isinstance(cfg, dict) and cfg.get("workflow_steps"):
            steps = _normalize_workflow_steps(cfg["workflow_steps"])
            task_name = task_name or cfg.get("name")
    if not steps:
        return {"messages": [SystemMessage(content="规划失败：缺少 workflow_steps，无法执行。")], "parent_action": "chat", "subgraph_delegated": False}

    task_name = task_name or skill_id or "动物数据分析"
    skill_config = {
        "name": task_name,
        "id": skill_id or "planned_workflow",
        "description": plan.get("reason") or "由父图规划生成的工作流",
        "type": "workflow",
        "parameters": {
            "work_dir": "工作目录路径", "input_file": "输入文件名",
            "history_file": "历史资料文件名", "pa": "居留型起始标记", "regional_level": "省级保护级别"
        },
        "workflow_steps": steps,
    }
    state_update.update({
        "execution_mode": "workflow",
        "selected_skill": task_name,
        "skill_config": skill_config,
        "workflow_steps": steps,
    })
    print(f"[DEBUG] parent_dispatch: 工作流 '{task_name}'，共 {len(steps)} 步")
    return state_update


def parent_dispatch_router(state: AgentState) -> str:
    if state.get("subgraph_delegated") is True:
        return "subgraph"
    return "parent_chat"


def parent_chat_node(state: AgentState):
    """父图聊天节点：回答一般问题 / 查看文件 / 汇报子图执行结果。"""
    docs = state.get("retrieved_docs") or []
    doc_block = ""
    if docs:
        doc_block = "\n\n检索到的相关资料：\n" + "\n\n".join(f"[资料{i + 1}] {d[:600]}" for i, d in enumerate(docs))
    messages = [SystemMessage(content=PARENT_SYSTEM_PROMPT + doc_block)] + state["messages"]
    response = parent_llm_with_tools.invoke(messages)

    if hasattr(response, 'tool_calls') and response.tool_calls:
        tc_summary = ", ".join(
            f"{tc['name']}({json.dumps(tc['args'], ensure_ascii=False)[:100]})" for tc in response.tool_calls)
        print(f"[DEBUG] parent_chat: LLM 调用工具 → {tc_summary}")
    elif response.content:
        print(f"[DEBUG] parent_chat: LLM 回复 → {response.content[:200]}")

    return {"messages": [response]}


def parent_chat_router(state: AgentState) -> str:
    last_message = state["messages"][-1] if state["messages"] else None
    if isinstance(last_message, AIMessage) and getattr(last_message, 'tool_calls', None):
        return "tools"
    return "respond"


def parent_plan_node(state: AgentState) -> Dict:
    """子图执行完成后的反馈节点：格式化结果，交回 parent_chat 汇报。"""
    final_output = state.get("final_output")

    if final_output:
        status = final_output.get("status", "unknown")
        skill_name = final_output.get("skill", "未知技能")
        if status == "completed":
            msg = f"✅ 子图任务 '{skill_name}' 执行完成。\n"
            if final_output.get("type") == "single":
                for r in final_output.get("results", []):
                    msg += f"  - {r.get('description', r.get('script', ''))}\n"
            else:
                for key, val in final_output.get("step_outputs", {}).items():
                    if isinstance(val, dict) and val.get("type") != "review":
                        files = val.get("output_files", [])
                        if files:
                            msg += f"  {key}: 输出文件 {files}\n"
        elif status == "error":
            msg = f"❌ 子图任务 '{skill_name}' 执行出错：\n{final_output.get('error', '未知错误')}"
        elif status == "aborted":
            msg = f"⚠️ 子图任务 '{skill_name}' 已终止：{final_output.get('reason', '用户终止')}"
        else:
            msg = f"子图任务 '{skill_name}' 状态：{status}"
        print(f"[DEBUG] parent_plan(feedback): {msg[:200]}")
        return {
            "execution_mode": "chat",
            "messages": [SystemMessage(content=msg)],
            "final_output": None,
            "subgraph_delegated": False,
        }

    print("[DEBUG] parent_plan: 子图返回但无 final_output")
    return {"execution_mode": "chat", "subgraph_delegated": False}


# ---------- 构建父图 ----------

parent_builder = StateGraph(AgentState)

parent_builder.add_node("parent_retrieve", parent_retrieve_node)
parent_builder.add_node("parent_planner", parent_planner_node)
parent_builder.add_node("parent_dispatch", parent_dispatch_node)
parent_builder.add_node("parent_chat", parent_chat_node)
parent_builder.add_node("parent_tools", parent_tool_node)
parent_builder.add_node("parent_plan", parent_plan_node)
parent_builder.add_node("subgraph", animal_date_analysis_agent)

parent_builder.add_edge(START, "parent_retrieve")
parent_builder.add_edge("parent_retrieve", "parent_planner")

parent_builder.add_conditional_edges("parent_planner", parent_planner_router, {
    "dispatch": "parent_dispatch",
    "chat": "parent_chat",
})

parent_builder.add_conditional_edges("parent_dispatch", parent_dispatch_router, {
    "subgraph": "subgraph",
    "parent_chat": "parent_chat",
})

parent_builder.add_conditional_edges("parent_chat", parent_chat_router, {
    "tools": "parent_tools",
    "respond": END,
})

parent_builder.add_edge("parent_tools", "parent_chat")
parent_builder.add_edge("subgraph", "parent_plan")
parent_builder.add_edge("parent_plan", "parent_chat")

# 编译父图
parent_graph = parent_builder.compile(checkpointer=checkpointer)


# ========== 主入口测试 ==========
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
            step_count = 0
            for event in parent_graph.stream(initial_state, config=config, stream_mode="values"):
                step_count += 1
                # 每个步骤打印当前消息流
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

            # 检查是否有 pending interrupt
            _handle_interrupts(parent_graph, config)

            # 输出最后一条 AI 消息
            _print_last_ai_message(parent_graph, config)

        except Exception as e:
            print(f"\n❌ 执行出错: {e}")


if __name__ == "__main__":
    run_interactive()

