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
└── persistent.py     # 持久化存储（JSON 文件）

demos/
└── chat_demo.py      # 交互式聊天 Demo
```

**核心概念**：
- **SimpleAgent**：支持同步/异步/流式对话
- **Memory**：对话历史管理，自动裁剪策略
- **System Prompt**：自定义 Agent 人设
- **Event System**：BEFORE_CHAT, AFTER_CHAT, STREAM_START/CHUNK/END
- **AgentConfig**：统一配置管理
- **PersistentMemory**：对话历史保存/加载

### 🔜 Phase 2 - 工具调用
```
core/
├── planner/         # 任务规划器
└── tools/           # 工具调用
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

### 2. 使用 CLI（单次对话）

```bash
# Ollama 本地模型（默认）
python -m cli "你好"

# 关闭思考模式（快速响应）
python -m cli --no-think "简单介绍一下 Python"

# 开启思考模式（更深入的推理）
python -m cli --think "什么是递归？"

# 指定模型
python -m cli --model llama3:8b "你好"

# 使用 API 提供者
python -m cli -p qwen -k sk-xxx "你好"
python -m cli -p glm -k xxx.xxx "写个快排"
python -m cli -p deepseek -k sk-xxx "解释一下量子计算"
```

### 3. 运行交互式 Demo（多轮对话）

```bash
python demos/chat_demo.py
```

选择 Provider（推荐 Ollama 本地），开始多轮对话。

### 4. 运行测试

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
    system_prompt="你是一个有用的助手"
)
agent = SimpleAgent(llm, config=config)

# 或直接传参（兼容旧方式）
agent = SimpleAgent(llm, max_memory_messages=100)
```

### 2. PersistentMemory 持久化

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

### 3. 事件系统

```python
from agents.simple import SimpleAgent, AgentEvent

# 注册事件处理器
agent.on(AgentEvent.BEFORE_CHAT, lambda **kwargs: print(f"Before: {kwargs.get('user_input')}"))
agent.on(AgentEvent.AFTER_CHAT, lambda response: print(f"Response: {response[:50]}..."))
agent.on(AgentEvent.STREAM_CHUNK, lambda chunk: print(chunk, end="", flush=True))

# 使用 Agent
response = agent.chat("Hello")
```

## 学习路线

- [x] **Phase 0** - LLM 抽象层
- [x] **Phase 1** - 最小 Agent（事件系统、配置类、持久化）
- [ ] **Phase 2** - 工具调用（Function Calling）
- [ ] **Phase 3** - Browser Agent（Playwright + DOM）
- [ ] **Phase 4** - 进阶（长期记忆、ReAct、反思）

## License

MIT
