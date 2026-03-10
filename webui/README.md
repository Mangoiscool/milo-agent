# Milo Agent Web UI

一个现代化的 Web 界面，用于与 Milo Agent 交互。

## 功能特性

- 🎨 现代化渐变 UI 设计
- 💬 实时 WebSocket 对话
- 🧩 工具调用可视化
- 🔌 多 LLM 提供者支持
- 📝 Markdown 渲染
- 📱 响应式设计

## 安装

```bash
pip install 'milo-agent[webui]'
# 或使用 conda
conda install -n milo-agent fastapi uvicorn websockets
```

## 启动

### 方法 1: 使用 CLI

```bash
python -m cli webui
```

### 方法 2: 直接运行脚本

```bash
python webui/launch.py
```

### 自定义端口和地址

```bash
python -m cli webui --host 0.0.0.0 --port 8080
```

### 开发模式（热重载）

```bash
python -m cli webui --reload
```

## 访问

启动后，在浏览器中打开：`http://localhost:8000`

## LLM 提供者

| 提供者 | 需要配置 | 工具调用 |
|--------|----------|----------|
| Ollama | 无需 API Key | ❌ |
| 通义千问 | `QWEN_API_KEY` | ✅ |
| 智谱 GLM | `GLM_API_KEY` | ✅ |
| DeepSeek | `DEEPSEEK_API_KEY` | ✅ |

## 可用工具

当使用支持工具调用的 API 提供者时，可以使用以下工具：

- 🧮 **Calculator** - 数学计算
- 🌤️ **Weather** - 天气查询
- 🔍 **WebSearch** - 网络搜索
- 📄 **FileRead** - 文件读取
- 💻 **CodeExecution** - 代码执行

## API 端点

### GET `/`
返回 Web UI 主页

### GET `/api/providers`
获取支持的 LLM 提供者列表

### POST `/api/agent/create`
创建新的 Agent 会话

```json
{
  "provider": "qwen",
  "api_key": "sk-xxx",
  "model": "qwen-plus"
}
```

### WebSocket `/ws/chat/{session_id}`
实时对话接口

## WebSocket 消息格式

### 客户端 → 服务器

```json
{
  "message": "你好"
}
```

### 服务器 → 客户端

```json
// 用户消息确认
{ "type": "user", "content": "你好" }

// Assistant 回复
{ "type": "assistant", "content": "..." }

// 工具调用
{ "type": "tool_call", "name": "calculator", "args": "{'a': 1, 'b': 2}" }

// 工具结果
{ "type": "tool_result", "name": "calculator", "result": "3", "is_error": false }

// 错误
{ "type": "error", "message": "..." }
```

## 项目结构

```
webui/
├── __init__.py      # 包初始化
├── server.py        # FastAPI 服务器
├── launch.py        # 启动脚本
├── static/
│   └── index.html   # 前端页面
└── README.md        # 本文档
```

## 技术栈

- **后端**: FastAPI + WebSocket
- **前端**: 原生 HTML/CSS/JavaScript
- **Markdown**: marked.js
- **ASGI 服务器**: Uvicorn
