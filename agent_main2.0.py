import os, json, subprocess, re
import sys
from docx import Document
from pathlib import Path
from typing import TypedDict, Optional, Any, List, Dict, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt, Command, Interrupt
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage, HumanMessage
from langchain_openai import ChatOpenAI


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


# ========== model ==========
llm = ChatOpenAI(
    model="qwen-plus",
    api_key=os.environ.get("DASHSCOPE_API_KEY"),
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
def run_python(script_path: str, keywords: str) -> str:
    """运行 python 脚本，传递关键词参数。

    Args:
        script_path: Python 脚本路径
        keywords: 命令行参数，空格分隔，如 "--work_dir ./data --input_file test.xlsx"
    """
    import subprocess as sp
    try:
        if not os.path.exists(script_path):
            return f"错误：脚本 {script_path} 不存在"
        kw_list = keywords.split() if keywords.strip() else []
        cmd = [sys.executable, script_path] + kw_list
        result = sp.run(cmd, capture_output=True, text=True, encoding='utf-8', timeout=120)
        if result.returncode != 0:
            return f"执行失败 (code {result.returncode}):\nSTDERR:\n{result.stderr}\nSTDOUT:\n{result.stdout}"
        return result.stdout
    except sp.TimeoutExpired:
        return "错误：脚本执行超时（120秒）"
    except Exception as e:
        return f"执行异常：{e}"


@tool
def run_r(script_path: str, keywords: str) -> str:
    """运行 R 脚本，传递关键词参数。

    Args:
        script_path: R 脚本路径
        keywords: 命令行参数，空格分隔，如 "--work_dir ./data --input_file test.xlsx"
    """
    try:
        if not os.path.exists(script_path):
            return f"错误：脚本 {script_path} 不存在"
        rscript = "Rscript" if os.name != "nt" else "Rscript.exe"
        kw_list = keywords.split() if keywords.strip() else []
        cmd = [rscript, script_path] + kw_list
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

    #print(f"步骤2开始前，{state.keys()} {state["step_outputs"]}")

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

    #print(f"新输入文件为{params["input_file"]}")

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

def substitute_params(keywords: List[str], params: Dict) -> List[str]:
    """将 keywords 中的 {var} 替换为 params 中的值"""
    result = []
    for kw in keywords:
        for key, val in params.items():
            kw = kw.replace(f"{{{key}}}", str(val))
        result.append(kw)

    return result

def _clean_unresolved_keywords(keywords: List[str]) -> List[str]:
    """成对移除未替换的可选参数：如 "--ref_file {ref_file}" → 两个都移除"""
    cleaned = []
    for kw in keywords:
        if '{' in kw and '}' in kw:
            # 这个值未被替换，同时移除它前面的 flag（如 --ref_file）
            if cleaned and cleaned[-1].startswith('--'):
                cleaned.pop()
            continue
        cleaned.append(kw)
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
    print(f"[DEBUG] execute_step: 步骤 {current_idx+1}/{len(steps)}, type={step_type}, detail={step}")

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
            keywords = substitute_params(wf_step["params"]["keywords"], params)
            keywords = _clean_unresolved_keywords(keywords)
            print(f"execute_skill keywords :{keywords}")
            kw_str = " ".join(keywords) if isinstance(keywords, list) else keywords
            print(f"[DEBUG] execute_skill: run_python {wf_step['params']['script_path']} args={kw_str}")
            result = run_python.invoke({"script_path": wf_step["params"]["script_path"], "keywords": kw_str})
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
            keywords = substitute_params(wf_step["params"]["keywords"], params)
            keywords = _clean_unresolved_keywords(keywords)
            kw_str = " ".join(keywords) if isinstance(keywords, list) else keywords
            print(f"[DEBUG] execute_skill: run_r {wf_step['params']['script_path']} args={kw_str}")
            result = run_r.invoke({"script_path": wf_step["params"]["script_path"], "keywords": kw_str})
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
        "content_structured": {       # 保留结构化数据供程序解析
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

        # 替换 keywords 中的模板变量（例如 {input_file1}）
        keywords = substitute_params(params.get("keywords", []), replace_vars)
        keywords = _clean_unresolved_keywords(keywords)
        kw_str = " ".join(keywords) if isinstance(keywords, list) else keywords

        print(f"[DEBUG] single_executor: 步骤{idx+1} tool={tool_name} script={script_path}")
        print(f"[DEBUG] single_executor: 命令行参数: {kw_str}")

        if tool_name == "run_python":
            result = run_python.invoke({"script_path": params["script_path"], "keywords": kw_str})
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
            result = run_r.invoke({"script_path": params["script_path"], "keywords": kw_str})
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
    """将执行结果转换为对话消息，回到 chat 模式，LLM 会分析并给建议"""
    output = state.get("final_output", {})
    status = output.get("status", "unknown")
    skill_name = output.get("skill", "未知技能")

    if status == "completed":
        msg = f"✅ 技能 '{skill_name}' 执行完成。\n"
        step_outputs = output.get("step_outputs", {})
        if step_outputs:
            for key, val in step_outputs.items():
                if isinstance(val, dict) and val.get("type") != "review":
                    files = val.get("output_files", [])
                    if files:
                        msg += f"  {key}: 输出文件 {files}\n"
        results = output.get("results", [])
        if results:
            for r in results:
                desc = r.get("description", "")
                success = r.get("success", True)
                status_icon = "✅" if success else "❌"
                msg += f"  {status_icon} {desc}\n"

    elif status == "error":
        error_detail = output.get("error", output.get("message", "未知错误"))
        msg = f"❌ 技能 '{skill_name}' 执行出错。\n\n**错误信息:**\n{error_detail}\n\n"

        # 拼接失败步骤的详细信息
        step_outputs = output.get("step_outputs", {})
        for key, val in step_outputs.items():
            if isinstance(val, dict) and val.get("results"):
                for r in val["results"]:
                    if not r.get("success", True):
                        msg += f"**失败脚本:** {r.get('script', '未知')}\n"
                        msg += f"**脚本描述:** {r.get('description', '无')}\n"
                        msg += f"**执行参数:** {json.dumps(val.get('params', {}), ensure_ascii=False)}\n"
                        msg += f"**错误输出:**\n{r.get('result', '')}\n\n"

        # 单技能模式的错误
        results = output.get("results", [])
        for r in results:
            if not r.get("success", True):
                msg += f"**失败脚本:** {r.get('script', '未知')}\n"
                msg += f"**脚本描述:** {r.get('description', '无')}\n"
                msg += f"**错误输出:**\n{r.get('result', '')}\n\n"

        msg += "请根据以上错误信息分析原因，向用户说明问题并提供修复建议。如果需要修改参数重新执行，请告诉用户。"

    elif status == "aborted":
        msg = f"⚠️ 技能 '{skill_name}' 已终止。原因: {output.get('reason', '用户终止')}"
    else:
        msg = f"技能 '{skill_name}' 执行结束，状态: {status}"

    return {
        "execution_mode": "chat",
        "messages": [SystemMessage(content=msg)]
    }


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


# ========== 构建图 ==========
builder = StateGraph(AgentState)

# --- Chat 层节点 ---
builder.add_node("chat", chat_node)
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
for i in range(20):
    builder.add_node(f"retry_{i}", lambda s, idx=i: retry_node_fn(s, idx))

# --- 边 ---
# 入口 → chat
builder.add_edge(START, "chat")

# chat 路由
builder.add_conditional_edges("chat", chat_router, {
    "tools": "tools",                  # 普通工具调用
    "tools_then_launch": "tools",      # launch_skill 也先过 ToolNode
    "prepare_launch": "prepare_launch", # 兜底，正常不会走这里
    "retry_with_correction": "correction",  # LLM 编造结果时纠正
    "respond": END,
})

# 纠正节点 → 回到 chat 让 LLM 重新调用 launch_skill
builder.add_edge("correction", "chat")

# tools 执行后路由
builder.add_conditional_edges("tools", after_tools_router, {
    "chat": "chat",
    "prepare_launch": "prepare_launch",
})

# prepare_launch → 根据 mode 路由
builder.add_conditional_edges("prepare_launch", mode_router, {
    "workflow": "executor",
    "single_skill": "single_executor",
    "chat": "chat",
})

# --- 工作流执行路由 ---
builder.add_conditional_edges("executor", workflow_router, {
    "workflow_complete": "complete",
    "workflow_abort": "abort",
    "execute_step": "executor",
    "continue_step": "continue",
    **{f"retry_step_{i}": f"retry_{i}" for i in range(20)}
})

builder.add_edge("continue", "executor")
for i in range(20):
    builder.add_edge(f"retry_{i}", "executor")

# 完成和终止 → 报告结果 → 回到 chat
builder.add_edge("complete", "report_results")
builder.add_edge("abort", "report_results")
builder.add_edge("single_executor", "report_results")

# 报告结果后回到 chat（LLM 会总结结果并告知用户）
builder.add_edge("report_results", "chat")


# ========== 编译 ==========
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
            for event in agent.stream(initial_state, config=config, stream_mode="values"):
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
                    # 完整消息流（每5个步骤打一次，避免刷屏）
                    if step_count % 5 == 0:
                        _log_messages(event, f"step-{step_count}")

            # 检查是否有 pending interrupt
            _handle_interrupts(agent, config)

            # 输出最后一条 AI 消息
            _print_last_ai_message(agent, config)

        except Exception as e:
            print(f"\n❌ 执行出错: {e}")


if __name__ == "__main__":
    run_interactive()
