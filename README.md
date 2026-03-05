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
    └── ollama.py     # 本地 Ollama
```

**核心概念**：
- **抽象层**：解耦业务逻辑和具体模型实现
- **Message**：统一的消息格式（role + content）
- **工厂模式**：`create_llm()` 一键创建不同模型

## 快速开始

### 1. 安装依赖

```bash
cd milo-agent
pip install -e .
```

### 2. 设置 API Key

```bash
# 选择一个或多个
export GLM_API_KEY="your-glm-api-key"
export QWEN_API_KEY="your-qwen-api-key"
export DEEPSEEK_API_KEY="your-deepseek-api-key"
```

### 3. 测试 LLM 模块

```bash
# 测试 GLM API
python tests/test_llm.py --provider glm

# 测试 Qwen API
python tests/test_llm.py --provider qwen

# 测试 Ollama 本地（需要先安装 Ollama 并下载模型）
python tests/test_llm.py --provider ollama --model qwen3.5:4b

# 测试流式输出
python tests/test_llm.py --provider glm --stream
```

## 项目结构

```
milo-agent/
├── core/                 # 核心模块
│   ├── llm/             # ✅ LLM 抽象层
│   ├── memory/          # 🔜 记忆系统
│   ├── planner/         # 🔜 任务规划器
│   └── tools/           # 🔜 工具调用
├── agents/
│   ├── browser/         # 🔜 Browser Agent
│   └── base.py          # 🔜 Agent 基类
├── skills/              # 🔜 技能模块
├── tests/               # 测试
└── config/              # 配置文件
```

## 学习路线

- [x] **Phase 0** - LLM 抽象层（当前）
- [ ] **Phase 1** - 最小 Agent（多轮对话 + 记忆）
- [ ] **Phase 2** - 工具调用（Function Calling）
- [ ] **Phase 3** - Browser Agent（Playwright + DOM）
- [ ] **Phase 4** - 进阶（长期记忆、ReAct、反思）

## License

MIT

Hello Agent!
