# 示例代码

本目录包含了 Milo Agent 的各种示例，按功能分类组织。

## 📚 目录结构

```
examples/
├── basic/                 # 基础功能示例
│   └── chat.py           # 交互式聊天演示
├── tools/                 # 工具使用示例
│   ├── tool_demo.py      # 工具调用演示
│   └── web_search.py     # 网络搜索工具示例
├── agents/                # Agent 类型示例
│   ├── rag_demo.py       # RAG Agent 演示
│   └── browser_demo.py   # Browser Agent 演示
└── advanced/              # 高级功能示例
    └── complete_agent.py  # 完整 Agent 功能演示
```

## 🚀 快速开始

### 1. 基础示例

```bash
# 交互式聊天（多轮对话）
python examples/basic/chat.py

# 选择 LLM 提供者并开始对话
```

### 2. 工具使用示例

```bash
# 工具调用演示
python examples/tools/tool_demo.py

# 网络搜索工具演示
python examples/tools/web_search.py
```

### 3. Agent 示例

```bash
# RAG Agent 演示（需要先拉取 embedding 模型）
ollama pull qwen3-embedding:0.6b
python examples/agents/rag_demo.py

# Browser Agent 演示（需要安装 playwright）
pip install playwright
playwright install chromium
python examples/agents/browser_demo.py
```

### 4. 高级示例

```bash
# 完整 Agent 功能演示（包含多种工具）
python examples/advanced/complete_agent.py
```

## 💡 运行说明

1. 确保已安装项目依赖：`pip install -e .`
2. 使用 Ollama 时确保本地服务正在运行
3. 使用 API 提供者时请设置相应的环境变量：
   ```bash
   export QWEN_API_KEY="your-api-key"
   export GLM_API_KEY="your-api-key"
   ```

## 📖 相关文档

- [README.md](../README.md) - 项目主文档
- [快速开始](../README.md#快速开始) - 基本使用方法