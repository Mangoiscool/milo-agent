# 示例代码

本目录包含了 Milo Agent 的示例，按学习顺序编号。

## 📚 文件列表

| 文件 | 说明 |
|------|------|
| `01_chat_demo.py` | 基础聊天 - 多轮对话、选择 LLM 提供者 |
| `02_tool_demo.py` | 工具调用 - 计算器、日期时间、随机数 |
| `03_web_search_demo.py` | 网络搜索 - DuckDuckGo 搜索工具 |
| `04_rag_agent_demo.py` | RAG Agent - 知识库问答 |
| `05_browser_agent_demo.py` | Browser Agent - 网页自动化 |

## 🚀 快速开始

```bash
# 1. 基础聊天（支持 qwen/glm/deepseek/ollama）
python examples/01_chat_demo.py

# 2. 工具调用演示（需要设置 QWEN_API_KEY）
export QWEN_API_KEY="your-api-key"
python examples/02_tool_demo.py

# 3. 网络搜索演示
python examples/03_web_search_demo.py

# 4. RAG Agent 演示（需要先拉取 embedding 模型）
ollama pull qwen3-embedding:0.6b
python examples/04_rag_agent_demo.py

# 5. Browser Agent 演示（需要安装 playwright）
pip install playwright && playwright install chromium
python examples/05_browser_agent_demo.py
```

## 💡 运行说明

1. 确保已安装项目依赖：`pip install -e .`
2. 使用 Ollama 时确保本地服务正在运行
3. 使用 API 提供者时请设置相应的环境变量：
   ```bash
   export QWEN_API_KEY="your-api-key"
   export GLM_API_KEY="your-api-key"
   ```