# Milo Agent

> 从零构建的通用型 AI Agent

## 项目简介

这是一个从零开始构建的 AI Agent 项目，目标是：
- 深入理解 Agent 的核心原理
- 支持多种 LLM（Qwen、GLM、DeepSeek、Ollama 本地）
- 构建 RAG Agent（本地知识库检索）
- 构建 Browser Agent（网页自动交互）

## 项目结构

```
milo-agent/
├── pyproject.toml              # 项目配置
├── .env.example               # 环境变量模板
├── cli/                       # 命令行工具
│   ├── __init__.py
│   └── main.py               # CLI 主入口
├── config/                    # 配置管理
│   ├── settings.yaml         # YAML 配置文件
│   └── settings.py          # 统一配置管理
├── core/                      # 核心基础设施
│   ├── __init__.py
│   ├── logger.py             # 日志模块
│   ├── structured_logger.py   # 结构化日志
│   ├── llm/                 # LLM 抽象层
│   │   ├── __init__.py
│   │   ├── base.py           # LLM 抽象基类
│   │   ├── factory.py        # 工厂方法
│   │   └── providers/
│   │       ├── api.py        # API 提供者（Qwen/GLM/DeepSeek）
│   │       └── ollama.py     # 本地 Ollama（支持思考模式）
│   ├── memory/              # 记忆系统
│   │   ├── __init__.py
│   │   ├── base.py           # 记忆系统抽象基类
│   │   ├── short_term.py     # 短期记忆（自动裁剪）
│   │   ├── scoring.py        # 消息评分系统
│   │   └── persistent.py     # 持久化存储（JSON 文件）
│   ├── tools/               # 工具调用
│   │   ├── __init__.py
│   │   ├── base.py           # 工具抽象基类
│   │   ├── registry.py       # 工具注册表
│   │   ├── retry.py          # 重试机制
│   │   ├── builtin/          # 内置工具
│   │   │   ├── calculator.py
│   │   │   ├── datetime.py
│   │   │   ├── random.py
│   │   │   ├── weather.py
│   │   │   ├── web_search.py
│   │   │   ├── file_operations.py
│   │   │   └── code_execution.py
│   │   ├── mcp.py           # MCP 协议支持
│   │   └── mcp_example.py   # MCP 使用示例
│   ├── rag/                 # RAG 模块
│   │   ├── __init__.py
│   │   ├── base.py           # RAG 基类
│   │   ├── document_loader.py # 文档加载器
│   │   ├── text_splitter.py  # 文本切分
│   │   ├── embeddings.py     # Embedding 抽象
│   │   ├── vector_store.py   # 向量存储
│   │   ├── retriever.py      # 检索器
│   │   └── tools.py          # RAG 工具
│   └── browser/              # Browser 模块
│       ├── __init__.py
│       ├── base.py           # Browser 基类
│       ├── controller.py     # 浏览器控制器
│       └── tools.py          # 浏览器工具
├── agents/                   # Agent 实现
│   ├── __init__.py
│   ├── base.py              # BaseAgent 基类
│   ├── agent_config.py      # AgentConfig 配置类
│   ├── simple.py            # SimpleAgent 实现
│   ├── main.py              # MainAgent - 统一主 Agent
│   ├── rag.py               # RAG Agent
│   └── browser.py           # Browser Agent
├── workspace/                # 工作目录
│   ├── knowledge_base/       # 知识库存储
│   └── browser_use/          # 浏览器截图
├── examples/                 # 示例代码
│   ├── 01_chat_demo.py      # 基础对话
│   ├── 02_tool_demo.py      # 工具调用
│   ├── 03_web_search_demo.py # 网络搜索
│   ├── 04_rag_agent_demo.py # RAG Agent
│   ├── 05_browser_agent_demo.py # Browser Agent
│   └── 06_main_agent_demo.py # MainAgent 完整演示
├── webui/                    # Web 界面
│   ├── server.py            # FastAPI Web 服务器
│   ├── launch.py            # 启动脚本
│   └── static/
│       └── index.html       # Web UI 前端
└── tests/                    # 测试代码
```

## 当前进度

### Phase 0 - LLM 抽象层
- 抽象基类（Message, Role, BaseLLM）
- 工厂方法（create_llm）
- API 提供者（Qwen, GLM, DeepSeek）
- 本地 Ollama（支持思考模式）

### Phase 1 - 最小 Agent
- SimpleAgent（多轮对话 + 记忆）
- 事件系统：扩展性基础
- 流式回退：自动降级机制
- AgentConfig：统一配置类
- PersistentMemory：持久化存储
- 消息评分系统：智能裁剪

### Phase 2 - 工具调用
- 工具抽象基类
- 工具注册表（带重试机制）
- 内置工具（9个）：calculator, datetime, random, weather, web_search, file_read, file_write, list_dir, code_execution
- MCP (Model Context Protocol) 支持
- Web UI 界面
- 结构化日志

### Phase 3 - RAG Agent & Browser Agent

#### Phase 3.1 - RAG 基础设施
- 文档加载器（PDF、Markdown、Word、Excel、图像）
- 文本切分器（RecursiveCharacterTextSplitter）
- Embedding 抽象层（本地 Ollama + API 提供商）
- 向量存储（ChromaDB，支持持久化）
- 检索器（余弦相似度 + MMR）

#### Phase 3.2 - RAG Agent
- RAG 工具：knowledge_search, knowledge_add, knowledge_list, knowledge_remove
- 多知识库管理
- 增量更新支持

#### Phase 3.3 - Browser Agent
- Playwright 集成
- 浏览器工具（8个）：navigate, click, type, scroll, get_text, screenshot, wait, back
- 截图保存到 workspace/browser_use/screenshots/

### Phase 4 - 进阶（规划中）
- 长期记忆（对话历史向量化、跨会话检索）
- ReAct 框架
- 反思机制
- 多 Agent 协作

## 快速开始

### 1. 环境准备

```bash
# 进入项目目录
cd milo-agent

# 激活虚拟环境
conda activate milo-agent

# 安装基础依赖
pip install -e .

# 安装可选依赖（按需）
pip install -e ".[webui]"  # Web UI
pip install -e ".[rag]"    # RAG 功能
pip install -e ".[browser]" # Browser 功能
pip install -e ".[dev]"    # 开发工具
```

### 2. 配置环境变量

```bash
# 复制配置模板
cp .env.example .env

# 编辑 .env 文件
vi .env
```

**关键配置项**：
- `DEFAULT_PROVIDER`: 默认 LLM 提供者（ollama/qwen/glm/deepseek）
- `QWEN_API_KEY` / `GLM_API_KEY` / `DEEPSEEK_API_KEY`: API 密钥

### 3. CLI 单次对话

```bash
# 使用 Ollama 本地模型（默认）
python -m cli.main "你好"

# 关闭思考模式（快速响应）
python -m cli.main --no-think "简单介绍一下 Python"

# 使用 API 提供者
python -m cli.main -p qwen "你好"
python -m cli.main -p glm "写个快排"
```

### 4. Web UI

```bash
# 启动 Web UI
python -m cli.main webui

# 自定义端口
python -m cli.main webui --port 8080
```

访问 `http://localhost:8000` 使用图形化界面。

**Web UI 功能**：
- 多模型切换（Ollama / 通义千问 / GLM / DeepSeek）
- RAG 能力开关
- Browser 能力开关
- 代码高亮
- 对话导出（Markdown/JSON/纯文本）

### 5. 运行示例

```bash
# 基础对话
python examples/01_chat_demo.py

# 工具调用
python examples/02_tool_demo.py

# RAG Agent
python examples/04_rag_agent_demo.py

# Browser Agent
python examples/05_browser_agent_demo.py

# MainAgent 完整演示
python examples/06_main_agent_demo.py
```

### 6. 运行测试

```bash
pytest tests/ -v
```

## 核心概念

### MainAgent - 统一入口

`MainAgent` 整合了所有能力，是最推荐的使用方式：

```python
from core.llm.factory import create_llm
from core.rag.embeddings import create_embedding
from agents.main import MainAgent

llm = create_llm("qwen", api_key="...")
embedding = create_embedding("ollama", model="nomic-embed-text")

# 基础使用（仅内置工具）
agent = MainAgent(llm)
response = agent.chat_with_tools("今天天气怎么样？")

# 启用 RAG
agent = MainAgent(
    llm=llm,
    enable_rag=True,
    embedding_model=embedding
)
agent.add_document("guide.pdf")
response = agent.chat_with_tools("文档里有什么？")

# 启用 Browser
agent = MainAgent(llm, enable_browser=True)
await agent.initialize()
response = agent.chat_with_tools("打开百度搜索 Python")
await agent.close()

# 完整功能
async with MainAgent(
    llm=llm,
    enable_rag=True,
    embedding_model=embedding,
    enable_browser=True
) as agent:
    agent.add_document("company_guide.pdf")
    response = agent.chat_with_tools("帮我查一下...")
```

### MainAgent 工具分类

| 类别 | 工具 | 说明 |
|------|------|------|
| **内置工具** | calculator, datetime, random, weather, web_search, file_read, file_write, list_dir, code_execution | 默认启用 |
| **RAG 工具** | knowledge_search, knowledge_add, knowledge_list, knowledge_remove | enable_rag=True |
| **Browser 工具** | browser_navigate, browser_click, browser_type, browser_scroll, browser_get_text, browser_screenshot, browser_wait, browser_back | enable_browser=True |

### 知识库管理 API

```python
agent = MainAgent(llm, enable_rag=True, embedding_model=embedding)

# 添加文档
agent.add_document("guide.pdf")
agent.add_text("自定义文本内容", source="user_input")
agent.add_directory("./docs", extensions=[".md", ".txt"])

# 查询知识库
sources = agent.list_sources()
stats = agent.get_knowledge_base_stats()

# 删除文档
agent.remove_document("guide.pdf")
```

### LLM 抽象层

```python
from core.llm.factory import create_llm

# 一行代码切换模型
llm = create_llm("qwen", api_key="xxx")
llm = create_llm("ollama", model="qwen3.5:4b", think=False)
```

### 记忆系统

```python
from core.memory.short_term import ShortTermMemory
from core.memory.persistent import PersistentMemory

# 短期记忆
memory = ShortTermMemory(max_messages=50, use_intelligent_pruning=True)

# 持久化存储
memory = PersistentMemory(
    max_messages=100,
    storage_path="./chat.json"
)
memory.save()
memory.load()
```

### 事件系统

```python
from agents.simple import SimpleAgent, AgentEvent

agent.on(AgentEvent.BEFORE_CHAT, lambda **kw: print(f"Before: {kw}"))
agent.on(AgentEvent.AFTER_CHAT, lambda r: print(f"Response: {r[:50]}"))
agent.on(AgentEvent.STREAM_CHUNK, lambda c: print(c, end="", flush=True))
```

## 学习路线

- [x] **Phase 0** - LLM 抽象层
- [x] **Phase 1** - 最小 Agent（事件系统、配置类、持久化）
- [x] **Phase 2** - 工具调用 + Web UI
- [x] **Phase 3** - RAG Agent & Browser Agent
  - [x] Phase 3.1 - RAG 基础设施
  - [x] Phase 3.2 - RAG Agent
  - [x] Phase 3.3 - Browser Agent
- [ ] **Phase 4** - 进阶（长期记忆、ReAct、反思、多 Agent 协作）

## License

MIT