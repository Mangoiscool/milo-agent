# Milo Agent 🎩

> 从零构建的通用型 AI Agent

## 项目简介

这是一个从零开始构建的 AI Agent 项目，目标是：
- 深入理解 Agent 的核心原理
- 支持多种 LLM（Qwen、GLM、Ollama 本地）
- 最终构建一个实用的 Browser Agent

## 当前进度

### ✅ Phase 0 - LLM 抽象层
```
core/llm/
├── base.py           # 抽象基类（Message, Role, BaseLLM）
├── factory.py        # 工厂方法（create_llm）
└── providers/
    ├── api.py        # API 提供者（Qwen, GLM, DeepSeek）
    └── ollama.py     # 本地 Ollama（支持思考模式）
```

### ✅ Phase 1 - 最小 Agent
```
agents/
└── simple.py         # SimpleAgent（多轮对话 + 记忆）
      - 事件系统：扩展性基础
      - 流式回退：自动降级机制
      - AgentConfig：统一配置类
      - PersistentMemory：持久化存储

core/memory/
├── base.py           # 记忆系统抽象基类
├── short_term.py     # 短期记忆（自动裁剪）
│   └── scoring.py   # 消息重要性评分系统（新增）
└── persistent.py     # 持久化存储（JSON 文件）

demos/
└── chat_demo.py      # 交互式聊天 Demo
```

**核心概念**：
- **SimpleAgent**：支持同步/异步/流式对话
- **Memory**：对话历史管理，自动裁剪策略（支持智能评分）
- **System Prompt**：自定义 Agent 人设
- **Event System**：BEFORE_CHAT, AFTER_CHAT, STREAM_START/CHUNK/END
- **AgentConfig**：统一配置管理
- **PersistentMemory**：对话历史保存/加载

### ✅ Phase 2 - 工具调用
```
core/tools/
├── base.py           # 工具抽象基类
├── registry.py       # 工具注册表（带重试机制）
│   └── retry.py     # 重试机制（新增）
├── builtin/
│   ├── calculator.py # 计算器
│   ├── weather.py    # 天气查询
│   ├── web_search.py # 网络搜索
│   ├── file_operations.py # 文件操作
│   └── code_execution.py # 代码执行
├── mcp.py           # MCP (Model Context Protocol) 支持
└── mcp_example.py   # MCP 使用示例

config/
└── settings.py       # 统一配置管理（新增）
.env.example          # 环境变量模板（新增）

webui/
├── server.py        # FastAPI Web 服务器
├── launch.py        # 启动脚本
└── static/
    └── index.html   # Web UI 前端（代码高亮 + 导出功能）

core/
└── structured_logger.py  # 结构化日志（新增）
```

### 🔜 Phase 3 - Browser Agent
```
agents/
└── browser/         # Playwright + DOM 操作
```

### 🔜 Phase 4 - 进阶
```
core/memory/
└── long_term/      # 长期记忆（向量存储）
```

## 快速开始

### 1. 安装依赖

```bash
cd milo-agent
pip install -e .
```

### 2. 配置环境变量（可选）

```bash
# 复制配置模板
cp .env.example .env

# 编辑 .env 文件，设置你的 API keys
vi .env
```

### 3. 使用 CLI（单次对话）

```bash
# Ollama 本地模型（默认）
python -m cli "你好"

# 关闭思考模式（快速响应）
python -m cli --no-think "简单介绍一下 Python"

# 开启思考模式（更深入的推理）
python -m cli --think "什么是递归？"

# 指定模型
python -m cli --model qwen3.5:4b "你好"

# 使用 API 提供者
python -m cli -p qwen -k sk-xxx "你好"
python -m cli -p glm -k xxx.xxx "写个快排"
python -m cli -p deepseek -k sk-xxx "解释一下量子计算"
```

### 4. Web UI 界面

启动 Web UI 服务器（需要安装额外依赖）：

```bash
# 安装 Web UI 依赖
pip install 'milo-agent[webui]'
# 或使用 conda
conda install -n milo-agent fastapi uvicorn websockets

# 启动 Web UI
python -m cli webui

# 自定义端口
python -m cli webui --port 8080
```

访问 `http://localhost:8000` 即可使用图形化界面。

**Web UI 新功能**：
- 🎨 代码高亮（Highlight.js）
- 📥 对话导出（Markdown/JSON/纯文本）
- 📋 一键复制代码块

### 5. 运行示例

#### 基础示例
```bash
# 交互式聊天 Demo
python examples/basic/chat.py

# 工具使用 Demo
python examples/tools/tool_demo.py
```

#### 高级示例
```bash
# 完整 Agent 功能演示
python examples/advanced/complete_agent.py

# 网络搜索工具演示
python examples/tools/web_search.py
```

### 6. 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 只测试 Phase 1
pytest tests/test_memory.py tests/test_simple_agent.py -v
```

## 核心概念

### 1. AgentConfig 统一配置

```python
from agents.config import AgentConfig
from agents.simple import SimpleAgent

# 使用配置类（推荐）
config = AgentConfig(
    enable_stream_fallback=False,
    max_memory_messages=100,
    system_prompt="你是一个有用的助手",
    use_intelligent_pruning=True  # 新增：启用智能裁剪
)
agent = SimpleAgent(llm, config=config)

# 或直接传参（兼容旧方式）
agent = SimpleAgent(llm, max_memory_messages=100)
```

### 2. 持久化存储

```python
from core.memory.persistent import PersistentMemory

# 持久化记忆
memory = PersistentMemory(max_messages=100)

# 保存到默认路径 (~/.milo-agent/memory.json)
memory.save()

# 加载记忆
count = memory.load()

# 自定义路径
memory = PersistentMemory(storage_path="./my_chat.json")
```

### 3. 智能消息裁剪

```python
from agents.config import AgentConfig

# 启用智能裁剪（基于消息重要性评分）
config = AgentConfig(use_intelligent_pruning=True)
agent = SimpleAgent(llm, config=config)
```

**评分因素**：
- 角色权重：SYSTEM > ASSISTANT > USER > TOOL
- 内容长度：较长的消息得分更高
- 时间衰减：最近消息得分更高
- 关键词：包含"错误"、"重要"等关键词加分

### 4. 工具调用重试机制

工具执行失败时自动重试，支持：
- 指数退避 + 随机抖动
- 可重试错误类型自动识别
- 可配置重试次数和延迟

```python
from core.tools.retry import RetryConfig

# 自定义重试配置
config = RetryConfig(
    max_retries=3,
    initial_delay=1.0,
    max_delay=30.0
)
```

### 5. 事件系统

```python
from agents.simple import SimpleAgent, AgentEvent

# 注册事件处理器
agent.on(AgentEvent.BEFORE_CHAT, lambda **kwargs: print(f"Before: {kwargs.get('user_input')}"))
agent.on(AgentEvent.AFTER_CHAT, lambda response: print(f"Response: {response[:50]}..."))
agent.on(AgentEvent.STREAM_CHUNK, lambda chunk: print(chunk, end="", flush=True))

# 使用 Agent
response = agent.chat("Hello")
```

### 6. 结构化日志

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

### 7. 环境变量配置

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

## 学习路线

- [x] **Phase 0** - LLM 抽象层
- [x] **Phase 1** - 最小 Agent（事件系统、配置类、持久化）
- [x] **Phase 2** - 工具调用（Function Calling）+ Web UI
- [ ] **Phase 3** - Browser Agent（Playwright + DOM）
- [ ] **Phase 4** - 进阶（长期记忆、ReAct、反思）

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

项目包含丰富的示例代码，按功能分类在 `examples/` 目录：

- **`examples/basic/`** - 基础功能示例
  - `chat.py` - 交互式聊天演示
- **`examples/tools/`** - 工具使用示例
  - `tool_demo.py` - 工具调用演示
  - `web_search.py` - 网络搜索工具示例
- **`examples/advanced/`** - 高级功能示例
  - `complete_agent.py` - 完整 Agent 功能演示

**Web UI**:
- **`webui/`** - Web 界面
  - `server.py` - FastAPI 服务器
  - `static/index.html` - 前端页面
  - [使用文档](webui/README.md)

## License

MIT
