# Phase 0 完成清单

## 📦 已创建的文件

```
milo-agent/
├── pyproject.toml              # 项目配置
├── README.md                   # 项目文档
├── GETTING_STARTED.md          # 快速开始指南
├── cli.py                      # ✨ 命令行工具
├── __init__.py
├── config/
│   └── settings.yaml           # 配置文件（支持环境变量）
├── core/
│   ├── __init__.py
│   └── llm/
│       ├── __init__.py
│       ├── base.py             # ✨ LLM 抽象基类
│       ├── factory.py          # ✨ 工厂方法
│       └── providers/
│           ├── api.py          # ✨ API 提供者（Qwen/GLM/DeepSeek）
│           └── ollama.py       # ✨ 本地 Ollama（支持思考模式）
└── tests/
    ├── __init__.py
    ├── test_factory.py         # ✨ 工厂方法测试
    ├── test_api_provider.py    # ✨ API 提供者测试
    └── test_ollama_provider.py # ✨ Ollama 提供者测试
```

## 🚀 快速测试

```bash
# 1. 进入项目目录
cd /path/to/milo-agent

# 2. 激活虚拟环境
conda activate milo-agent

# 3. 安装依赖（如果还没安装）
pip install -e .

# 4. 设置 API Key（API 提供者需要）
export GLM_API_KEY="your-key"
# 或
export QWEN_API_KEY="your-key"
```

### 使用 Ollama 本地模型

```bash
# 前提：已安装并启动 Ollama，拉取模型
ollama pull qwen3.5:4b

# 运行 CLI（默认使用 Ollama）
python -m cli "你好"

# 关闭思考模式（快速响应）
python -m cli --no-think "简单介绍一下 Python"

# 开启思考模式（更深入的推理）
python -m cli --think "什么是递归？"

# 指定模型
python -m cli --model llama3:8b "你好"
```

### 使用 API 提供者

```bash
# Qwen
python -m cli -p qwen -k sk-xxx "你好"

# GLM
python -m cli -p glm -k xxx.xxx "写个快排"

# DeepSeek
python -m cli -p deepseek -k sk-xxx "解释一下量子计算"
```

### CLI 参数说明

| 参数 | 简写 | 说明 |
|------|------|------|
| `--provider` | `-p` | LLM 提供者 (qwen/glm/deepseek/ollama) |
| `--model` | `-m` | 模型名称 |
| `--api-key` | `-k` | API 密钥 |
| `--think` | | 开启思考模式 |
| `--no-think` | | 关闭思考模式 |
| `--temperature` | `-t` | 温度参数 |
| `--debug` | `-d` | 启用调试日志 |

## 💡 核心学习点

### 1. 为什么需要抽象层？
- 业务代码不依赖具体模型
- 切换模型只需改配置
- 统一错误处理、重试逻辑

### 2. Message 设计
```python
Message(role=Role.USER, content="你好")
# → {"role": "user", "content": "你好"}
```
角色划分让模型理解对话结构（system/user/assistant）

### 3. 工厂模式
```python
# 一行代码切换模型
llm = create_llm("glm", api_key="xxx")
llm = create_llm("ollama", model="qwen3.5:4b", think=False)
```

### 4. 思考模式（Think Mode）
Ollama Qwen3 等模型支持思考模式：
- **开启** (`think=True`)：模型先在内部推理，再输出最终答案（质量更高，速度较慢）
- **关闭** (`think=False`)：直接输出答案（速度快，适合简单问题）

配置方式：
```yaml
# config/settings.yaml
ollama:
  think: false  # 关闭思考模式
```

```python
# 代码中指定
llm = create_llm("ollama", think=False)
```

```bash
# 命令行指定
python -m cli --no-think "你好"
```

### 5. 同步 vs 异步
- 同步：简单场景，测试用
- 异步：Agent 并发调用工具时必需

### 6. 流式输出
- 提升用户体验（不用等完整回复）
- 减少首字延迟感知

## 📝 下一步

Phase 1 将实现：
- Agent 基类
- 对话历史管理
- 简单的记忆系统

---

继续学习，有问题随时问 🎩
