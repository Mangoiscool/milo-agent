# milo-agent 快速开始

## 📦 项目结构

```
milo-agent/
├── pyproject.toml              # 项目配置
├── README.md                   # 项目文档
├── GETTING_STARTED.md          # 本文档
├── cli.py                      # ✨ 命令行工具
├── __init__.py
├── config/
│   └── settings.yaml           # 配置文件（支持环境变量）
├── core/
│   ├── __init__.py
│   ├── logger.py               # ✨ 日志模块
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base.py             # ✨ LLM 抽象基类
│   │   ├── factory.py          # ✨ 工厂方法
│   │   └── providers/
│   │       ├── api.py          # ✨ API 提供者（Qwen/GLM/DeepSeek）
│   │       └── ollama.py       # ✨ 本地 Ollama（支持思考模式）
│   └── memory/
│       ├── __init__.py
│       ├── base.py             # ✨ 记忆系统抽象基类
│       ├── short_term.py       # ✨ 短期记忆（自动裁剪）
│       └── persistent.py       # ✨ 持久化存储（JSON 文件）
├── agents/
│   ├── __init__.py
│   ├── config.py              # ✨ AgentConfig 配置类
│   └── simple.py               # ✨ SimpleAgent 实现
├── demos/
│   └── chat_demo.py            # ✨ 交互式聊天 Demo
└── tests/
    ├── __init__.py
    ├── test_factory.py         # ✨ 工厂方法测试
    ├── test_llm_base.py        # ✨ LLM 基类测试
    ├── test_llm.py             # ✨ LLM 集成测试
    ├── test_api_provider.py    # ✨ API 提供者测试
    ├── test_ollama_provider.py # ✨ Ollama 提供者测试
    ├── test_memory.py          # ✨ 记忆系统测试
    └── test_simple_agent.py    # ✨ Agent 测试
```

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

### 2. CLI 单次对话

```bash
# 使用 Ollama 本地模型（默认）
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

### 3. 交互式 Demo（多轮对话）

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

### 4. 运行测试

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
from agents.config import AgentConfig

llm = create_llm("ollama", model="qwen3.5:4b")

# 方式1：使用配置类（推荐）
config = AgentConfig(
    enable_stream_fallback=True,
    max_memory_messages=100,
    system_prompt="你是一个有用的助手"
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
memory = ShortTermMemory(max_messages=50)

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

### 5. AgentConfig 统一配置

```python
from agents.config import AgentConfig

config = AgentConfig(
    enable_stream_fallback=False,   # 关闭流式回退
    max_memory_messages=100,      # 最大消息数
    system_prompt="You are helpful"  # 系统提示词
)

agent = SimpleAgent(llm, config=config)
```

### 6. PersistentMemory 持久化存储

```python
from core.memory.persistent import PersistentMemory

memory = PersistentMemory(max_messages=100)
memory.save()  # 保存到 ~/.milo-agent/memory.json
memory.load()  # 从文件加载
```

**特性**：
- 继承 ShortTermMemory，保留所有原有功能
- `save()`: 保存到 JSON 文件
- `load()`: 从文件加载，返回消息数
- 添加消息时自动保存
- 清空时删除存储文件

### 7. 事件系统

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
```

### 8. 流式回退机制

```python
# 自动启用（默认）
agent = SimpleAgent(llm)

# 手动关闭
agent = SimpleAgent(llm, enable_stream_fallback=False)
```

当流式 API 失败时，自动回退到异步聊天模式，保证可用性。

### 9. 思考模式（Think Mode）

Qwen3 等模型支持思考模式：
- **开启** (`think=True`)：模型先内部推理，再输出答案（质量高，速度慢）
- **关闭** (`think=False`)：直接输出答案（速度快）

```python
llm = create_llm("ollama", think=False)  # 代码中指定
```

```bash
python -m cli --no-think "你好"  # CLI 指定
```

## 📝 下一步

Phase 2 将实现：
- Function Calling 工具调用
- 工具注册与发现
- Agent 自动选择工具

---

继续学习，有问题随时问 🎩
