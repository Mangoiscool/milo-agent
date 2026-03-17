# Milo Agent 🎩

> 从零构建的通用型 AI Agent

## 项目简介

这是一个从零开始构建的 AI Agent 项目，目标是：
- 深入理解 Agent 的核心原理
- 支持多种 LLM（Qwen、GLM、Ollama 本地）
- 构建 RAG Agent（本地知识库检索）
- 构建 Browser Agent（网页自动交互）

## 📦 项目结构

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
│   └── tools/               # 工具调用
│       ├── __init__.py
│       ├── base.py           # 工具抽象基类
│       ├── registry.py       # 工具注册表
│       ├── retry.py          # 重试机制
│       ├── builtin/          # 内置工具
│       │   ├── calculator.py
│       │   ├── weather.py
│       │   ├── web_search.py
│       │   ├── file_operations.py
│       │   └── code_execution.py
│       ├── mcp.py           # MCP 协议支持
│       └── mcp_example.py   # MCP 使用示例
│   └── rag/                 # RAG 模块
│       ├── __init__.py
│       ├── base.py           # RAG 基类
│       ├── document_loader.py # 文档加载器
│       ├── text_splitter.py  # 文本切分
│       ├── embeddings.py     # Embedding 抽象
│       ├── vector_store.py   # 向量存储
│       └── retriever.py      # 检索器
│   └── browser/              # Browser 模块
│       ├── __init__.py
│       ├── base.py           # Browser 基类
│       ├── controller.py     # 浏览器控制器
│       └── tools.py          # 浏览器工具
├── agents/                   # Agent 实现
│   ├── __init__.py
│   ├── agent_config.py      # AgentConfig 配置类
│   ├── simple.py           # SimpleAgent 实现
│   ├── rag.py              # RAG Agent
│   └── browser.py          # Browser Agent
├── knowledge_base/           # 知识库目录
│   └── .gitkeep
├── examples/                 # 示例代码
│   ├── basic/               # 基础示例
│   │   └── chat.py
│   ├── tools/               # 工具示例
│   │   ├── tool_demo.py
│   │   └── web_search.py
│   ├── agents/              # Agent 示例
│   │   ├── rag_demo.py
│   │   └── browser_demo.py
│   └── advanced/            # 高级示例
│       └── complete_agent.py
├── webui/                    # Web 界面
│   ├── server.py            # FastAPI Web 服务器
│   ├── launch.py            # 启动脚本
│   └── static/
│       └── index.html       # Web UI 前端
├── tests/                    # 测试代码
│   ├── __init__.py
│   ├── test_factory.py
│   ├── test_llm_base.py
│   ├── test_llm.py
│   ├── test_api_provider.py
│   ├── test_ollama_provider.py
│   ├── test_memory.py
│   └── test_simple_agent.py
└── skills/                   # 技能模块（预留）
    └── README.md
```

## 当前进度

### ✅ Phase 0 - LLM 抽象层
- 抽象基类（Message, Role, BaseLLM）
- 工厂方法（create_llm）
- API 提供者（Qwen, GLM, DeepSeek）
- 本地 Ollama（支持思考模式）

### ✅ Phase 1 - 最小 Agent
- SimpleAgent（多轮对话 + 记忆）
- 事件系统：扩展性基础
- 流式回退：自动降级机制
- AgentConfig：统一配置类
- PersistentMemory：持久化存储
- 消息评分系统：智能裁剪

### ✅ Phase 2 - 工具调用
- 工具抽象基类
- 工具注册表（带重试机制）
- 内置工具：计算器、天气查询、网络搜索、文件操作、代码执行
- MCP (Model Context Protocol) 支持
- Web UI 界面
- 结构化日志

### 🔜 Phase 3 - RAG Agent & Browser Agent

#### Phase 3.1 - RAG 基础设施
- 文档加载器（PDF、Markdown、Word、Excel、图像）
- 文本切分器（RecursiveCharacterTextSplitter）
- Embedding 抽象层（本地 Ollama + API 提供商）
- 向量存储（ChromaDB，支持持久化）
- 检索器（余弦相似度 + MMR）

#### Phase 3.2 - RAG Agent
- 继承 SimpleAgent，集成检索能力
- 多知识库管理（创建、更新、删除）
- 增量更新支持
- Web UI 集成（知识库管理界面）

#### Phase 3.3 - Browser Agent
- Playwright 集成
- DOM 操作与元素交互
- 网页自动浏览和数据提取

### 🔜 Phase 4 - 进阶
- 长期记忆（对话历史向量化、跨会话检索）
- ReAct 框架
- 反思机制
- 多 Agent 协作

## 🚀 快速开始

### 1. 环境准备

```bash
# 进入项目目录
cd milo-agent

# 激活虚拟环境
conda activate milo-agent

# 安装依赖
pip install -e .
```

### 2. 配置环境变量（推荐）

```bash
# 复制配置模板
cp .env.example .env

# 编辑 .env 文件，设置你的 API keys
vi .env
```

**推荐配置项**：
- `DEFAULT_PROVIDER`: 选择默认的 LLM 提供者
- `QWEN_API_KEY` / `GLM_API_KEY` / `DEEPSEEK_API_KEY`: API 密钥
- `MAX_MEMORY_MESSAGES`: 最大消息数量
- `USE_INTELLIGENT_PRUNING`: 是否启用智能裁剪
- `USE_STRUCTURED_LOGGING`: 是否使用 JSON 格式日志

### 3. CLI 单次对话

```bash
# 使用 Ollama 本地模型（默认）
python -m cli.main "你好"

# 关闭思考模式（快速响应）
python -m cli.main --no-think "简单介绍一下 Python"

# 开启思考模式（更深入的推理）
python -m cli.main --think "什么是递归？"

# 指定模型
python -m cli.main --model qwen3.5:4b "你好"

# 使用 API 提供者（推荐使用 .env 配置）
python -m cli.main -p qwen "你好"
python -m cli.main -p glm "写个快排"
python -m cli.main -p deepseek "解释一下量子计算"
```

### 4. 交互式 Demo（多轮对话）

```bash
python demos/chat_demo.py
```

按提示选择：
1. **Provider**: 选 `4` (Ollama) - 本地运行，无需 API key
2. **Model**: 默认 `qwen3.5:4b`
3. **Mode**: 选 `2` (流式输出)

**支持命令**：
- `history` - 查看对话历史
- `clear` - 清空记忆
- `quit` - 退出

### 5. Web UI 界面

启动 Web UI 服务器：

```bash
# 启动 Web UI
python -m cli.main webui

# 自定义端口
python -m cli.main webui --port 8080
```

访问 `http://localhost:8000` 即可使用图形化界面。

**Web UI 功能**：
- 🎨 代码高亮（Highlight.js）
- 📥 对话导出（Markdown/JSON/纯文本）
- 📋 一键复制代码块

### 6. 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 只测试 Phase 1（Agent + Memory）
pytest tests/test_memory.py tests/test_simple_agent.py tests/test_event_system.py -v
```

## 💡 核心概念

### 1. LLM 抽象层

```python
from core.llm.factory import create_llm

# 一行代码切换模型
llm = create_llm("glm", api_key="xxx")
llm = create_llm("ollama", model="qwen3.5:4b", think=False)
```

### 2. Message 设计

```python
from core.llm.base import Message, Role

Message(role=Role.USER, content="你好")
# → {"role": "user", "content": "你好"}
```

### 3. SimpleAgent

```python
from core.llm.factory import create_llm
from agents.simple import SimpleAgent
from agents.agent_config import AgentConfig

llm = create_llm("ollama", model="qwen3.5:4b")

# 方式1：使用配置类（推荐）
config = AgentConfig(
    enable_stream_fallback=True,
    max_memory_messages=100,
    system_prompt="你是一个有用的助手",
    use_intelligent_pruning=True  # 启用智能裁剪
)
agent = SimpleAgent(llm, config=config)

# 方式2：直接传参（兼容旧方式）
agent = SimpleAgent(llm, max_memory_messages=100)

# 同步对话
response = agent.chat("你好！")

# 异步对话
response = await agent.achat("你好！")

# 流式输出
async for chunk in agent.astream("你好！"):
    print(chunk, end="", flush=True)

# 查看历史
history = agent.get_history()

# 清空记忆
agent.clear_history()
```

### 4. 记忆系统

```python
from core.memory.short_term import ShortTermMemory
from core.memory.persistent import PersistentMemory

# 短期记忆（内存中，程序结束丢失）
memory = ShortTermMemory(max_messages=50, use_intelligent_pruning=True)

# 持久化存储（保存到文件，重启后可恢复）
memory = PersistentMemory(
    max_messages=100,
    storage_path="./my_chat.json"
)

# 保存到文件
memory.save()

# 从文件加载
memory.load()
```

**PersistentMemory 特性**：
- 继承 ShortTermMemory，保留所有原有功能
- `save()`: 保存到 JSON 文件
- `load()`: 从文件加载，返回消息数
- 添加消息时自动保存
- 清空时删除存储文件

### 5. AgentConfig 统一配置

```python
from agents.agent_config import AgentConfig

config = AgentConfig(
    enable_stream_fallback=False,   # 关闭流式回退
    max_memory_messages=100,       # 最大消息数
    system_prompt="You are helpful",  # 系统提示词
    use_intelligent_pruning=True   # 启用智能裁剪
)

agent = SimpleAgent(llm, config=config)
```

### 6. 智能消息裁剪

```python
# 启用智能裁剪（基于消息重要性评分）
config = AgentConfig(use_intelligent_pruning=True)
agent = SimpleAgent(llm, config=config)
```

**评分因素**：
- 角色权重：SYSTEM > ASSISTANT > USER > TOOL
- 内容长度：较长的消息得分更高
- 时间衰减：最近消息得分更高
- 关键词：包含"错误"、"重要"等关键词加分

### 7. 工具调用重试

```python
from core.tools.retry import RetryConfig

# 自定义重试配置
retry_config = RetryConfig(
    max_retries=3,
    initial_delay=1.0,
    max_delay=30.0
)

# 使用自定义配置
from core.tools import ToolRegistry
registry = ToolRegistry(retry_config=retry_config)
```

**重试机制特性**：
- 指数退避 + 随机抖动
- 可重试错误类型自动识别
- 可配置重试次数和延迟

### 8. 事件系统

```python
from agents.simple import SimpleAgent, AgentEvent

# 注册事件处理器
agent.on(AgentEvent.BEFORE_CHAT, lambda **kwargs: print(f"Before: {kwargs}"))
agent.on(AgentEvent.AFTER_CHAT, lambda response: print(f"Response: {response[:50]}"))
agent.on(AgentEvent.STREAM_CHUNK, lambda chunk: print(chunk, end="", flush=True))

# 支持的事件类型
# BEFORE_CHAT      - 发送输入前
# AFTER_CHAT       - 接收回复后
# STREAM_START     - 流式开始
# STREAM_CHUNK     - 每个流块
# STREAM_END       - 流式结束
# MEMORY_PRUNED   - 记忆裁剪时
# TOOL_CALL        - 工具调用时
# TOOL_RESULT      - 工具返回结果时
```

### 9. 结构化日志

```python
# 在 .env 中启用
USE_STRUCTURED_LOGGING=true

# 或在代码中
from config.settings import settings
settings().use_structured_logging = True

# 使用结构化日志记录上下文
from core.structured_logger import get_structured_logger

logger = get_structured_logger("MyModule")
logger_with_context = logger.bind(session_id="abc123", user_id="user_456")

logger_with_context.info("Action completed", action="login", duration=0.123)
```

### 10. 环境变量配置

项目支持通过 `.env` 文件管理配置：

```bash
# LLM 配置
DEFAULT_PROVIDER=ollama
QWEN_API_KEY=sk-xxx
GLM_API_KEY=xxx.xxx

# Agent 配置
MAX_MEMORY_MESSAGES=50
USE_INTELLIGENT_PRUNING=false

# 日志配置
LOG_LEVEL=INFO
USE_STRUCTURED_LOGGING=false

# 工具配置
TOOL_MAX_RETRIES=3
```

### 11. 流式回退机制

```python
# 自动启用（默认）
agent = SimpleAgent(llm)

# 手动关闭
agent = SimpleAgent(llm, enable_stream_fallback=False)
```

当流式 API 失败时，自动回退到异步聊天模式，保证可用性。

### 12. 思考模式（Think Mode）

Qwen3 等模型支持思考模式：
- **开启** (`think=True`)：模型先内部推理，再输出答案（质量高，速度慢）
- **关闭** (`think=False`)：直接输出答案（速度快）

```python
llm = create_llm("ollama", think=False)  # 代码中指定
```

```bash
python -m cli.main --no-think "你好"  # CLI 指定
```

### 13. MCP (Model Context Protocol)

项目支持 MCP 协议，允许与各种工具和服务集成：

```python
from core.tools.mcp import MCPClient

# 连接到 MCP 服务器
client = MCPClient("http://localhost:3000")

# 获取可用工具
tools = client.list_tools()

# 调用工具
result = client.call_tool("calculator", expression="2+2")
```

## 学习路线

- [x] **Phase 0** - LLM 抽象层
- [x] **Phase 1** - 最小 Agent（事件系统、配置类、持久化）
- [x] **Phase 2** - 工具调用（Function Calling）+ Web UI
- [x] **Phase 3** - RAG Agent & Browser Agent
  - [x] Phase 3.1 - RAG 基础设施（文档加载、Embedding、向量存储）
  - [x] Phase 3.2 - RAG Agent（多知识库管理、增量更新）
  - [x] Phase 3.3 - Browser Agent（Playwright + DOM）
- [ ] **Phase 4** - 进阶（长期记忆、ReAct、反思、多 Agent 协作）

## 新增功能（v0.2.0）

### 🧠 内存优化
- 智能消息裁剪：基于重要性评分，保留关键信息
- 评分系统：综合考虑角色、长度、时间、关键词等因素

### 🔄 错误处理
- 工具调用重试机制：指数退避 + 随机抖动
- 可重试错误自动识别：网络超时、连接错误、速率限制等

### ⚙️ 配置管理
- `.env` 文件支持：统一环境变量管理
- 配置模板：`.env.example` 提供完整配置示例
- 多环境支持：开发、测试、生产环境配置分离

### 📝 日志优化
- 结构化日志：JSON 格式，便于日志收集和分析
- 上下文绑定：记录会话 ID、用户 ID 等上下文信息
- 灵活配置：支持文本/JSON 两种格式切换

### 🎨 Web UI 增强
- 代码高亮：集成 Highlight.js，支持多种编程语言
- 对话导出：支持 Markdown、JSON、纯文本三种格式
- 复制按钮：代码块一键复制

## 示例代码

项目包含丰富的示例代码：

### 基础示例
```bash
# 交互式聊天 Demo
python examples/basic/chat.py
```

### 工具使用示例
```bash
# 工具调用 Demo
python examples/tools/tool_demo.py

# 网络搜索工具演示
python examples/tools/web_search.py
```

### Agent 示例
```bash
# RAG Agent 演示
python examples/agents/rag_demo.py

# Browser Agent 演示
python examples/agents/browser_demo.py

# 完整功能演示
python examples/advanced/complete_agent.py
```

### Web UI
```bash
# 启动 Web 服务器
python -m cli.main webui
```

访问 `http://localhost:8000` 即可使用图形化界面。

## License

MIT
