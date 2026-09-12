# EcoAgent · 生态环境调查数据分析智能体

基于 [LangGraph](https://github.com/langchain-ai/langgraph) 的多智能体（多图）系统，面向陆生动物 / 陆生植物调查与生态环境评估场景（样线法调查、历史资料整理、物种名录生成、多样性分析、报告撰写等），自动完成：

> **混合检索 → 任务路由 → 领域子图规划 → 脚本执行（Python/R）→ 人工审核中断 → 结果汇报**

核心设计思想（agent3.5）：

- **父图不再承担"规划器"职责**，只负责：接收用户消息 → 【需求判断】（动物任务 / 植物任务 / 知识问答） 
- → 数据分析/报告任务：将【原始用户输入原样】转发给对应子图 → 接收子图结果汇报；
- → 知识问答：从多个知识库中选择最匹配的一个做混合检索 → 基于检索资料简述回答。
- **规划器下沉到领域子图**：建立两个 base work agent（`animal_base_work_agent` 动物子图、`plant_base_work_agent` 植物子图）。子图收到任务后，先由各自的【任务规划器】根据「用户输入 + 本子图技能目录」输出 JSON，决定执行方式（`workflow` / `single_skill` / `chat`）。
- **状态分离**：子图内部执行字段（`execution_mode` / `workflow_steps` / `step_outputs` …）与父图完全隔离；父图与子图之间仅通过「桥接通道」交换最小信息：`animal_request / animal_result`（动物）、`plant_request / plant_result`（植物），互不覆盖。

---

## ✨ 功能特性

- **三图协作架构（父图 + 两个领域子图）**
  - 父图（协调层）：【需求判断】（动物任务 / 植物任务 / 知识问答）
  - - → 数据分析/报告任务：将【原始用户输入原样】转发给对应子图 → 接收子图结果汇报；
  - - → 知识问答：从多个知识库中选择最匹配的一个做混合检索 → 基于检索资料简述回答。
  - 动物子图 `animal_base_work_agent`：任务规划器 → 工作流/单技能执行器 → 审核中断 → 结果写回
  - 植物子图 `plant_base_work_agent`：与动物子图同一套内部结构（命名空间一致、状态分离），对应植物技能目录
- **混合检索（语义 + 关键词）**
  - 语义检索：DashScope `text-embedding-v3`
  - 关键词检索：纯 Python 实现的 Okapi BM25（中文整词 + 字符 bigram）
  - 融合算法：RRF（Reciprocal Rank Fusion），向量库使用 Chroma
- **技能热插拔**：技能以 JSON 声明式配置，新增技能无需改代码
  - 动物技能放在 `skills/` 根目录；植物技能放在 `plant_skills/` 目录（各自独立发现）
- **工作流编排 + 人工审核中断**：支持多步骤工作流、步骤间文件依赖、`interrupt()` 人工审核点、重试/终止；脚本执行失败也会触发中断等待人工决定
- **混合脚本执行**：同一工作流中可混用 Python 与 R 脚本
- **参数自动抽取**：子图规划器从自然语言中抽取工作路径、输入文件、保护级别、居留型等参数；用户未提供的键省略（脚本使用默认值，不编造文件路径）

---

## 🏗 架构设计

```
                          ┌────────────────────────────────────────────────┐
                          │                 父图 parent_graph               │
  用户输入 ───────────────▶│                                                │
                          │  parent_retrieve   混合检索 SOP 知识库            │
                          │        │                                       │
                          │        ▼                                       │
                          │  parent_router     任务路由（只分类，不规划）      │
                          │    ├─ animal ──┐                                │
                          │    ├─ plant  ──┤                                │
                          │    └─ chat ────┼──▶ parent_chat（问答/查看文件）   │
                          │        │       │        ▲                       │
                          │        ▼       │        │                       │
                          │  parent_dispatch│        │ 汇报（SystemMessage）   │
                          │   （原样转发用户输入到桥接通道） │                  │
                          └───────┬────────┴────────┼───────────────────────┘
                                  │                 │
                 animal_request   │                 │  plant_request
                                  ▼                 ▼
       ┌─────────────────────────────────┐  ┌─────────────────────────────────┐
       │   animal_base_work_agent（动物）  │  │   plant_base_work_agent（植物）   │
       │                                 │  │                                 │
       │  planner     任务规划器（输出JSON）│  │  planner     任务规划器（输出JSON） │
       │    ├─ workflow      ──▶ executor │  │    ├─ workflow      ──▶ executor │
       │    ├─ single_skill  ──▶ single_executor │    ├─ single_skill ──▶ ... │
       │    └─ need_info/chat ─▶ finish   │  │    └─ need_info/chat ─▶ finish   │
       │                                 │  │                                 │
       │  executor     多步骤工作流执行     │  │  executor     多步骤工作流执行     │
       │    ├─ skill 步骤（Python/R 脚本）  │  │    ├─ skill 步骤（Python/R 脚本）  │
       │    ├─ review 步骤（interrupt 审核）│  │    ├─ review 步骤（interrupt 审核）│
       │    └─ 执行失败中断（retry/abort）  │  │    └─ 执行失败中断（retry/abort）  │
       │                                 │  │                                 │
       │  finish      结果写回 animal_result│  │  finish      结果写回 plant_result │
       └────────────────┬────────────────┘  └────────────────┬────────────────┘
                        │                                      │
                        ▼                                      ▼
              父图 parent_report（汇总子图结果）→ parent_chat 汇报给用户
```

### 各图职责

| 图 | 节点 | 职责 |
|----|------|------|
| 父图 | `parent_retrieve` | 从 SOP 向量库混合检索相关资料 |
| 父图 | `parent_router` | 路由判断：`animal` / `plant` / `chat`（**不规划执行细节**） |
| 父图 | `parent_dispatch` | 将【原始用户输入原样】写入对应桥接通道 `animal_request` / `plant_request` |
| 父图 | `parent_report` | 汇总 `animal_result` / `plant_result`，格式化后交给对话层汇报 |
| 父图 | `parent_chat` | 回答一般问题、查看文件（`parent_list_files` / `parent_read_file_content`）、汇报子图执行结果 |
| 子图（动物/植物） | `planner` | 任务规划器：根据用户输入 + 本子图技能目录输出 JSON（`workflow` / `single_skill` / `chat`） |
| 子图（动物/植物） | `executor` | 多步骤工作流执行，支持步骤间文件依赖、review 审核中断、失败重试 |
| 子图（动物/植物） | `single_executor` | 单技能直接执行 |
| 子图（动物/植物） | `finish` | 收尾：把结果写回 `animal_result` / `plant_result`，并清空请求通道 |

### 状态分离与桥接通道

- **父图状态（`ParentState`）**：只包含对话 `messages`、检索结果 `retrieved_docs`、路由结果 `route / route_reason`，以及 4 个桥接通道 `animal_request / animal_result / plant_request / plant_result`。**不含任何子图执行字段。**
- **子图状态（`AnimalAnalysisState` / `PlantAnalysisState`）**：
  - 与父图同名的桥接通道（`animal_request / animal_result` 或 `plant_request / plant_result`）；
  - 共享对话通道 `messages`（父图/子图可见，子图执行路径不写 `messages`）；
  - 内部执行字段（`execution_mode` / `workflow_steps` / `step_outputs` …），父图不可见。
- 两个子图内部字段命名一致（命名空间可以相同），因为它们各自独立编译、checkpoint 命名空间隔离，互不干扰。动物结果落在 `animal_result`，植物结果落在 `plant_result`，互不覆盖。

---

## 🧩 技能系统

技能以 JSON 声明式配置，自动发现、无需改代码。动物技能放在 `skills/`，植物技能放在 `plant_skills/`。

### 动物技能（`skills/`）

| 技能 | id | 类型 | 说明 | 实现 |
|------|-----|------|------|------|
| 动物列表合并 | `animal_simple_list` | single | 合并样线表与历史资料中的动物中文名，去重并优先保留现场调查数据 | Python |
| 动物名录生成 | `animal_catalog_generate` | single | 对陆生动物物种数据初步整理，输出物种名录 | Python |
| 动物名录分析 | `animal_catalog_analysis` | single | 分析动物名录的目/科/种、区系、居留型等 | R |
| alpha多样性分析 | `alpha_diversity_analysis` | single | 基于样线表做 α 物种多样性分析 | R |
| 动物分析，报告撰写全工作流 | `full_animal_workflow` | workflow | 完整流程：列表合并 → 名录生成 → 审核 → 多样性分析 | 编排 |

### 植物技能（`plant_skills/`）

> ⚠️ 植物技能暂未创建/验证。`plant_base_work_agent` 的规划器与执行器已就绪，后续只需在 `plant_skills/` 下放入植物技能 JSON（及对应脚本）即可自动启用，无需改动代码。

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
- 子图规划器在用户说「全工作流 / 完整流程 / 报告撰写」时，优先复用目录中 `type=workflow` 技能的 `workflow_steps`；否则按用户意图把多个 single 技能按顺序拼接成工作流

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
python agent3.5.py
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

**子图规划器生成的工作流**

```
步骤1: skill → 动物列表合并
步骤2: skill → 动物名录生成（依赖步骤1输出）
步骤3: review → 人工审核步骤2结果
步骤4: skill → alpha多样性分析
```

**执行流程**

```
parent_retrieve    → 混合检索 SOP 相关资料
parent_router      → route=animal（只分类，不规划）
parent_dispatch    → 将原始用户输入原样写入 animal_request
animal_base_work_agent.planner → action=workflow，生成 4 步工作流
animal_base_work_agent.executor → 步骤1 ✅ 输出 动物列表.xlsx
                                → 步骤2 ✅ 输出 动物名录.xlsx
                                → 步骤3 ⏸ 人工审核中断，等待「通过/重新执行/终止」
parent_report      → 汇总 animal_result
parent_chat        → 向用户汇报结果
```

### 示例 2：知识问答（直接回答，不执行分析）

**输入**

```
鹈鹕的生活习性是什么
```

**流程**

```
parent_retrieve → 混合检索到 SOP 中的鹈鹕资料
parent_router   → route=chat（一般问答，即使提到动物名也归 chat）
parent_chat     → 结合检索资料直接回答
```

### 示例 3：植物任务（路由到植物子图）

**输入**

```
植物名录生成，工作路径：...，植物样线表：...
```

**流程**

```
parent_router → route=plant
parent_dispatch → 原样写入 plant_request
plant_base_work_agent.planner → 按 plant_skills/ 技能目录规划并执行
parent_report → 汇总 plant_result 汇报
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
| `EMBED_BASE_URL` | `` | 向量化接口地址 |
| `SOP_DOCX_PATH` | `D:\PythonProject1\生态环境调查报告工作sop.docx` | SOP 知识库文档路径 |
| `CHROMA_DIR` | 脚本目录下 `chroma_sop_db` | Chroma 向量库持久化目录 |

### 模型

- LLM 默认使用 DeepSeek 兼容接口（代码中为 `deepseek-v4-flash`，可在 `llm` / `parent_llm` 处修改为你的模型名）
- 语义向量化默认使用 DashScope `text-embedding-v3`

---

## 📁 目录结构

```
.
├── agent3.5.py                     # 主程序（父图 + 动物/植物两个领域子图）
├── skills/                         # 动物技能目录（JSON 声明式配置 + 脚本）
│   ├── animal_simple_list.json     #   动物列表合并（Python）
│   ├── animal_simple_list.py
│   ├── animal_catalog_generate.json#   动物名录生成（Python）
│   ├── animal_excel_catalog_generate.py
│   ├── animal_catalog_analysis.json#   动物名录分析（R）
│   ├── animal_catalog_analysis.R
│   ├── alpha_analysis.json         #   alpha多样性分析（R）
│   ├── Alpha_analysis.R
│   ├── full_animal_workflow.json   #   全工作流编排
│   └── 参考名录.xlsx               #   参考名录数据（示例数据，可用PostgreSQL数据库）
├── plant_skills/                   # 植物技能目录（待创建/验证，规划器与执行器已就绪）
├── chroma_sop_db/                  # Chroma 向量库（运行时生成）
├── requirements.txt                # Python 依赖
├── .env.example                    # 环境变量示例
└── README.md
```

---

## 📄 License

GPL v3

---

## 🙏 致谢

- [LangGraph](https://github.com/langchain-ai/langgraph) — 图/多智能体编排
- [LangChain](https://github.com/langchain-ai/langchain) — LLM 工具与消息抽象
- [Chroma](https://github.com/chroma-core/chroma) — 向量数据库
- [DashScope](https://dashscope.aliyun.com/) — 语义向量化
- [DeepSeek](https://www.deepseek.com/) — 大语言模型
- [Python](https://www.python.org/) — 主程序与脚本执行语言
- [R](https://www.r-project.org/) — 数据分析脚本语言（名录分析、多样性分析等）
