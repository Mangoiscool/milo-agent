# 示例代码

本目录包含了 Milo Agent 的示例，按学习顺序编号。

## 🏗️ Agent 架构

Milo Agent 采用统一的架构设计：

```
BaseAgent (抽象基类)
├── 核心能力：LLM、Memory、ToolRegistry
├── 对话接口：chat/achat/astream
└── 工具调用循环

├── SimpleAgent - 基础对话 Agent
├── ReActAgent - 推理与行动 Agent（显式思考过程）
├── MainAgent - 统一的 Main Agent（推荐）
│   ├── 内置工具（默认启用）
│   ├── RAG 能力（可选）
│   └── Browser 能力（可选）
├── RAGAgent - 知识库问答 Agent
└── BrowserAgent - 浏览器自动化 Agent
```

## 📚 文件列表

| 文件 | 说明 |
|------|------|
| `01_chat_demo.py` | 基础聊天 - 多轮对话、选择 LLM 提供者 |
| `02_tool_demo.py` | 工具调用 - 计算器、日期时间、随机数 |
| `03_web_search_demo.py` | 网络搜索 - DuckDuckGo 搜索工具 |
| `04_rag_agent_demo.py` | RAG Agent - 知识库问答 |
| `05_browser_agent_demo.py` | Browser Agent - 网页自动化 |
| `06_main_agent_demo.py` | **MainAgent - 统一的 Agent（推荐）** |
| `07_react_agent_demo.py` | ReAct Agent - 推理与行动（显式思考过程）|

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

# 6. MainAgent 演示（统一的 Agent）
python examples/06_main_agent_demo.py

# 7. ReAct Agent 演示（推理与行动）
python examples/07_react_agent_demo.py
```

## 💡 运行说明

1. 确保已安装项目依赖：`pip install -e .`
2. 使用 Ollama 时确保本地服务正在运行
3. 使用 API 提供者时请设置相应的环境变量：
   ```bash
   export QWEN_API_KEY="your-api-key"
   export GLM_API_KEY="your-api-key"
   ```

## 🎯 推荐使用 MainAgent

MainAgent 是统一的 Agent 实现，可以组合多种能力：

```python
from agents import MainAgent
from core.llm.factory import create_llm
from core.rag import create_embedding

llm = create_llm("qwen", api_key="...")
embedding = create_embedding("ollama", model="nomic-embed-text")

# 创建具备所有能力的 Agent
agent = MainAgent(
    llm=llm,
    enable_rag=True,           # 启用知识库
    embedding_model=embedding,
    enable_browser=True        # 启用浏览器
)

# 添加知识
agent.add_document("company_guide.pdf")

# 对话
response = agent.chat_with_tools("帮我查一下公司的请假流程")
```

### MainAgent 可用工具

| 类别 | 工具 | 说明 |
|------|------|------|
| **内置** | calculator | 计算器 |
| | datetime | 日期时间 |
| | web_search | 网络搜索 |
| | file_read/write | 文件操作 |
| | code_execution | 代码执行 |
| **RAG** | knowledge_search | 检索知识库 |
| | knowledge_add | 添加文档 |
| | knowledge_list | 列出文档来源 |
| | knowledge_remove | 移除文档 |
| **Browser** | browser_navigate | 导航 |
| | browser_click | 点击 |
| | browser_type | 输入 |
| | browser_screenshot | 截图 |

## 🧠 ReAct Agent

ReAct Agent 提供显式的思考过程（Reasoning + Acting）：

```python
from agents import ReActAgent
from core.llm.factory import create_llm

llm = create_llm("qwen", api_key="...")
agent = ReActAgent(llm=llm, tools=[CalculatorTool(), WeatherTool()])

# 显示思考过程
response = agent.chat("北京今天气温是多少？明天降温5度后呢？", show_reasoning=True)

# 输出示例：
# Thought: 用户询问北京气温和计算...
# Action: weather(city="北京")
# Observation: 晴天，25°C
# Thought: 今天25度，明天降温5度...
# Action: calculator(expression="25 - 5")
# Observation: 20
# Final Answer: 北京今天25°C，明天降温5度后是20°C
```

### ReAct vs ToolAgent

| 特性 | ToolAgent | ReActAgent |
|------|-----------|------------|
| 工具调用 | ✅ 直接调用 | ✅ 显式思考后调用 |
| 思考过程 | ❌ 隐藏 | ✅ 可追踪 |
| 多步骤推理 | ✅ 自动 | ✅ 显式步骤 |
| 调试友好 | ⚠️ 一般 | ✅ 非常友好 |