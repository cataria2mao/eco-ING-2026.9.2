# EcoAgent · 野生动物调查数据分析智能体

基于 [LangGraph](https://github.com/langchain-ai/langgraph) 的多智能体（多图）系统，面向野生动物调查与生态环境评估场景（样线法调查、历史资料整理、物种名录生成、多样性分析等），自动完成：

> **知识检索 → 工作流规划 → 子图执行（Python/R 脚本）→ 人工审核中断 → 结果汇报**

核心设计思想：**父图负责"理解需求 + 检索资料 + 规划工作流"，子图负责"发现技能 + 收集参数 + 执行脚本 + 中断审核"**。两图共享状态、完全解耦。

---

## ✨ 功能特性

- **双图（父图/子图）协作架构**
  - 父图（协调层）：混合检索 SOP 知识库 → LLM 规划器生成工作流计划 → 直接分发子图执行
  - 子图（执行层）：技能发现、参数收集、工作流编排、脚本执行、人工审核
- **混合检索（语义 + 关键词）**
  - 语义检索：DashScope `text-embedding-v3`
  - 关键词检索：纯 Python 实现的 Okapi BM25（中文整词 + 字符 bigram）
  - 融合算法：RRF（Reciprocal Rank Fusion），向量库使用 Chroma
- **技能热插拔**：技能以 JSON 声明式配置，新增技能无需改代码
- **工作流编排 + 人工审核中断**：支持多步骤工作流、步骤间文件依赖、`interrupt()` 人工审核点、重试/终止
- **混合脚本执行**：同一工作流中可混用 Python 与 R 脚本
- **参数自动抽取与校验**：LLM 从自然语言中抽取工作路径、输入文件、保护级别等参数，并对必填参数做校验

---

## 🏗 架构设计

```
                         ┌─────────────────────────────────────────────┐
                         │                  父图 parent_graph            │
                         │                                             │
  用户输入 ──────────────▶│ parent_retrieve  混合检索 SOP 知识库          │
                         │        │                                    │
                         │        ▼                                    │
                         │ parent_planner  LLM 规划器                   │
                         │   ├─ workflow / single_skill ──┐            │
                         │   └─ chat ────────────────────┐│            │
                         │        │                     ││            │
                         │        ▼                     │▼            │
                         │ parent_dispatch              parent_chat    │
                         │   （转换为子图执行状态）        （问答/汇报/文件）│
                         │        │                     │             │
                         └────────┼─────────────────────┴─────────────┘
                                  │ 直接分发（跳过子图 chat 层）
                                  ▼
                         ┌─────────────────────────────────────────────┐
                         │              子图 animal_date_analysis_agent  │
                         │                                             │
                         │  executor（工作流执行器）                     │
                         │    ├─ 步骤1: skill 动物列表合并               │
                         │    ├─ 步骤2: skill 动物名录生成（依赖步骤1输出）│
                         │    ├─ 步骤3: review 人工审核（interrupt）      │
                         │    └─ 步骤4: skill alpha多样性分析            │
                         │  single_executor（单技能执行器）              │
                         │  chat 层（技能发现/参数收集，可独立使用）       │
                         └─────────────────────────────────────────────┘
                                  │
                                  ▼
                         父图 parent_plan → parent_chat 汇报结果给用户
```

### 两图职责

| 图 | 节点 | 职责 |
|----|------|------|
| 父图 | `parent_retrieve` | 从 SOP 向量库混合检索相关资料 |
| 父图 | `parent_planner` | LLM 规划：判断 `workflow` / `single_skill` / `chat`，生成 JSON 计划 |
| 父图 | `parent_dispatch` | 将计划转换为子图可直接执行的状态（`execution_mode` + `workflow_steps` + 参数） |
| 父图 | `parent_chat` | 回答一般问题、查看文件、汇报子图执行结果 |
| 父图 | `parent_plan` | 格式化子图执行结果，交回 `parent_chat` 汇报 |
| 子图 | chat 层 | 技能发现（`get_skill_info`）、参数收集（`launch_skill`），可独立作为对话助手 |
| 子图 | `executor` | 多步骤工作流执行，支持步骤间文件依赖与 review 中断 |
| 子图 | `single_executor` | 单技能直接执行 |

---

## 🧩 技能系统

技能以 JSON 声明式配置，存放在 `skills/` 目录，自动发现、无需改代码。

| 技能 | id | 类型 | 说明 | 实现 |
|------|-----|------|------|------|
| 动物列表合并 | `animal_simple_list` | single | 合并样线表与历史资料中的动物中文名，去重并优先保留现场调查数据 | Python |
| 动物名录生成 | `animal_catalog_generate` | single | 对陆生动物物种数据初步整理，输出物种名录 | Python |
| 动物名录分析 | `animal_catalog_analysis` | single | 分析动物名录的目/科/种、区系、居留型等 | R |
| alpha多样性分析 | `alpha_diversity_analysis` | single | 基于样线表做 α 物种多样性分析 | R |
| 动物分析，报告撰写全工作流 | `full_animal_workflow` | workflow | 完整流程：列表合并 → 名录生成 → 审核 → 多样性分析 | 编排 |

### 技能 JSON 结构示例

```jsonc
{
  "name": "动物列表合并",
  "id": "animal_simple_list",
  "description": "合并样线表和历史资料中的动物中文名列表",
  "type": "single",              // single | workflow
  "parameters": {
    "work_dir": "工作目录路径，包含输入和输出文件",
    "input_file": "样线表文件名，如：样线表.xlsx",
    "history_file": "历史资料文件名，如：历史资料.xlsx",
    "output_file": "输出文件名，如：动物列表.xlsx"
  },
  "workflow": [                   // single 技能的脚本执行步骤
    {
      "step": 1,
      "tool": "run_python",       // run_python | run_r
      "params": {
        "script_path": "skills/animal_simple_list.py",
        "script_args": ["--work_dir", "{work_dir}", "--input_file", "{input_file}"]
      }
    }
  ]
}
```

### workflow 型技能（工作流编排）

```jsonc
{
  "name": "动物分析，报告撰写全工作流",
  "id": "full_animal_workflow",
  "type": "workflow",
  "parameters": {
    "work_dir": "工作目录路径",
    "input_file": "样线表文件名",
    "history_file": "历史资料文件名",
    "pa": "居留型起始标记（可选，默认 ⅤA）",
    "regional_level": "省级保护级别（可选）"
  },
  "workflow_steps": [
    { "step": 1, "type": "skill", "use_skill": "动物列表合并" },
    { "step": 2, "type": "skill", "use_skill": "动物名录生成", "input_from": 1 },
    { "step": 3, "type": "review", "description": "动物名录生成 结果" },
    { "step": 4, "type": "skill", "use_skill": "alpha多样性分析", "input_from": 2 }
  ]
}
```

- `input_from`：声明对上游步骤输出文件的依赖（自动把上一步的输出文件传给下一步）
- `type: review`：人工审核步骤，执行到这里会 `interrupt()` 暂停，等待用户选择「通过 / 重新执行 / 终止」

---

## 🚀 快速开始

### 1. 环境要求

- Python ≥ 3.10（本项目在 3.14 上验证通过）
- 运行 R 技能需要安装 [R](https://www.r-project.org/) 并确保 `Rscript` 在 PATH 中
- 可选：`git`、`pip`

### 2. 安装依赖

```bash
git clone <your-repo-url>
cd PythonProject1
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. 配置环境变量

复制 `.env.example` 为 `.env`，填写你的 API Key：

```bash
# DeepSeek LLM
DEEPSEEK_API_KEY=sk-xxxx
DEEPSEEK_BASE_URL=https://api.deepseek.com

# DashScope 向量化（语义检索）
DASHSCOPE_API_KEY=sk-xxxx
```

> ⚠️ 不要将真实的 `.env` 提交到仓库！`.env` 已在 `.gitignore` 中排除。

### 4. 准备 SOP 知识库（可选）

将你的工作 SOP 文档（Word 格式）放到指定路径，默认读取：

```
生态环境调查报告工作sop.docx
```

路径可用环境变量 `SOP_DOCX_PATH` 覆盖。首次检索时会自动读取分块、构建 Chroma 语义索引。

### 5. 运行

```bash
python agent3.2.py
```

进入交互式命令行后输入任务即可，例如：

```
🧑 你: 动物分析，报告撰写全工作流, 工作路径：D:\EcoAgentProject\广西风电鸟类监测，样线表文件：广西风电样线表.xlsx, 历史资料：广西风电动物历史资料.xlsx，pa：ⅤA，省级保护级别：广西自治区级
```

---

## 📖 使用示例

### 示例 1：完整工作流（从样线表开始）

**输入**

```
动物分析，报告撰写全工作流, 工作路径：D:\EcoAgentProject\广西风电鸟类监测，样线表文件：广西风电样线表.xlsx, 历史资料：广西风电动物历史资料.xlsx，pa：ⅤA，省级保护级别：广西自治区级
```

**父图规划出的工作流**

```
步骤1: skill → 动物列表合并
步骤2: skill → 动物名录生成（依赖步骤1输出）
步骤3: review → 人工审核步骤2结果
步骤4: skill → alpha多样性分析
```

**执行流程**

```
parent_retrieve   → 检索到 SOP 相关资料
parent_planner    → action=workflow, 生成 4 步工作流
parent_dispatch   → 直接分发子图执行
子图 executor     → 步骤1 ✅ 输出 动物列表.xlsx
                   → 步骤2 ✅ 输出 动物名录.xlsx
                   → 步骤3 ⏸ 人工审核中断，等待「通过/重试/终止」
```

### 示例 2：续接工作流（已有动物列表）

**输入**

```
动物报告撰写全工作流，已有动物列表，完成后续任务，工作路径：D:\EcoAgentProject\广西风电鸟类监测，动物列表文件：广西风电动物列表.xlsx, pa：ⅤA，省级保护级别：广西自治区级
```

**父图规划出的工作流**（自动跳过「动物列表合并」）

```
步骤1: skill → 动物名录生成
步骤2: review → 人工审核步骤1结果
步骤3: skill → alpha多样性分析（依赖步骤1输出）
```

### 示例 3：知识问答（直接回答，不执行分析）

**输入**

```
鹈鹕的生活习性是什么
```

**流程**

```
parent_retrieve → 混合检索到 SOP 中的鹈鹕资料
parent_planner  → action=chat
parent_chat     → 结合检索资料直接回答
```

---

## ⚙️ 配置说明

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DEEPSEEK_API_KEY` | — | DeepSeek LLM API Key |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | DeepSeek 兼容接口地址 |
| `DASHSCOPE_API_KEY` | — | DashScope 向量化 API Key |
| `EMBED_MODEL` | `text-embedding-v3` | 语义向量模型 |
| `EMBED_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | 向量化接口地址 |
| `SOP_DOCX_PATH` | `生态环境调查报告工作sop.docx` | SOP 知识库文档路径 |
| `CHROMA_DIR` | 脚本目录下 `chroma_sop_db` | Chroma 向量库持久化目录 |

### 模型

- LLM 默认使用 DeepSeek 兼容接口（代码中为 `deepseek-v4-flash`，可在 `llm` / `parent_llm` 处修改为你的模型名）
- 语义向量化默认使用 DashScope `text-embedding-v3`

---

## 📁 目录结构

```
.
├── agent3.2.py                     # 主程序（父图 + 子图，双图架构）
├── skills/                         # 技能目录（JSON 声明式配置 + 脚本）
│   ├── animal_simple_list.json     #   动物列表合并（Python）
│   ├── animal_simple_list.py
│   ├── animal_catalog_generate.json#   动物名录生成（Python）
│   ├── animal_excel_catalog_generate.py
│   ├── animal_catalog_analysis.json#   动物名录分析（R）
│   ├── animal_catalog_analysis.R
│   ├── alpha_analysis.json         #   alpha多样性分析（R）
│   ├── alpha_analysis.R
│   ├── full_animal_workflow.json   #   全工作流编排
│   └── 参考名录.xlsx               #   参考名录数据（请勿开源）
├── chroma_sop_db/                  # Chroma 向量库（运行时生成）
├── requirements.txt                # Python 依赖
├── .env.example                    # 环境变量示例
└── README.md
```

---

## 🔐 开源前注意事项

1. **API Key**：`.env` 中的 `DEEPSEEK_API_KEY`、`DASHSCOPE_API_KEY`、`LANGSMITH_API_KEY` 等属于敏感信息，**切勿提交**。
2. **业务数据**：样线表、历史资料、`参考名录.xlsx` 等属于项目/公司业务数据，建议**不要开源**，仅保留空样例或脱敏数据。
3. **本地运行产物**：`chroma_sop_db/`、`__pycache__/`、`.venv/`、`日志.docx`、`日志.md` 等运行时/日志文件不应入库。
4. 建议首次提交前执行 `git rm -r --cached .env chroma_sop_db`（若已误提交）。

仓库已附带 `.gitignore`，请按需调整。

---

## 📄 License

（请根据你的需求选择，例如 MIT / Apache-2.0，并在仓库根目录添加 `LICENSE` 文件。）

---

## 🙏 致谢

- [LangGraph](https://github.com/langchain-ai/langgraph) — 图/多智能体编排
- [LangChain](https://github.com/langchain-ai/langchain) — LLM 工具与消息抽象
- [Chroma](https://github.com/chroma-core/chroma) — 向量数据库
- [DashScope](https://dashscope.aliyun.com/) — 语义向量化
- [DeepSeek](https://www.deepseek.com/) — 大语言模型
