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

core/memory/
├── base.py           # 记忆系统抽象基类
└── short_term.py     # 短期记忆（自动裁剪）

demos/
└── chat_demo.py      # 交互式聊天 Demo
```

**核心概念**：
- **SimpleAgent**：支持同步/异步/流式对话
- **Memory**：对话历史管理，自动裁剪策略
- **System Prompt**：自定义 Agent 人设

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

# 开启思考模式
python -m cli --think "什么是递归？"

# 使用 API 提供者
python -m cli -p glm -k your-api-key "写个快排"
```

### 3. 运行交互式 Demo（多轮对话）

```bash
python demos/chat_demo.py
```

选择 Provider（推荐 Ollama 本地），开始多轮对话。

## 项目结构

```
milo-agent/
├── core/                 # 核心模块
│   ├── llm/             # ✅ LLM 抽象层
│   ├── memory/          # ✅ 记忆系统
│   ├── logger.py        # ✅ 日志模块
│   ├── planner/         # 🔜 任务规划器
│   └── tools/           # 🔜 工具调用
├── agents/
│   ├── simple.py        # ✅ 最小 Agent
│   └── browser/         # 🔜 Browser Agent
├── demos/
│   └── chat_demo.py     # ✅ 交互式聊天 Demo
├── skills/              # 🔜 技能模块
├── tests/               # ✅ 单元测试
└── config/              # 配置文件
```

## 学习路线

- [x] **Phase 0** - LLM 抽象层
- [x] **Phase 1** - 最小 Agent（多轮对话 + 记忆）
- [ ] **Phase 2** - 工具调用（Function Calling）
- [ ] **Phase 3** - Browser Agent（Playwright + DOM）
- [ ] **Phase 4** - 进阶（长期记忆、ReAct、反思）

## License

MIT
