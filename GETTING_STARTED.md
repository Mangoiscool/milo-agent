# Phase 0 完成清单

## 📦 已创建的文件

```
milo-agent/
├── pyproject.toml              # 项目配置
├── README.md                   # 项目文档
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
│           └── ollama.py       # ✨ 本地 Ollama
└── tests/
    ├── __init__.py
    └── test_llm.py             # 测试脚本
```

## 🚀 快速测试

```bash
# 1. 进入项目目录
cd ~/.openclaw/workspace/milo-agent

# 2. 安装依赖
pip install -e .

# 3. 设置 API Key（任选其一）
export GLM_API_KEY="your-key"

# 4. 运行测试
python tests/test_llm.py --provider glm

# 5. 测试流式输出
python tests/test_llm.py --provider glm --stream
```

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

### 4. 同步 vs 异步
- 同步：简单场景，测试用
- 异步：Agent 并发调用工具时必需

### 5. 流式输出
- 提升用户体验（不用等完整回复）
- 减少首字延迟感知

## 📝 下一步

Phase 1 将实现：
- Agent 基类
- 对话历史管理
- 简单的记忆系统

---

继续学习，有问题随时问 🎩
