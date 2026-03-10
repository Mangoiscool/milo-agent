# 快速开始指南

## 本地 Ollama 使用（推荐）

### 1. 安装 Ollama
```bash
# 访问 https://ollama.ai 下载并安装
# 或使用 Homebrew
brew install ollama
```

### 2. 下载模型
```bash
ollama pull qwen3.5:4b
```

### 3. 启动服务
```bash
ollama serve
```

### 4. 运行示例
```bash
# 交互模式（Ollama 不支持工具调用）
python examples/advanced/complete_agent.py --provider ollama

# 或直接对话
python examples/advanced/complete_agent.py --provider ollama "你好"
```

## API 使用（需要 API key）

### 1. 设置环境变量
```bash
export QWEN_API_KEY='your-api-key'
# 或
export GLM_API_KEY='your-api-key'
```

### 2. 运行示例（支持工具调用）
```bash
python examples/advanced/complete_agent.py --provider qwen
```

## 特点对比

| 特性 | Ollama | API 提供者 |
|------|--------|-----------|
| 工具调用 | ❌ 不支持 | ✅ 支持 |
| 本地部署 | ✅ 是 | ❌ 否 |
| 隐私保护 | ✅ 数据不出本地 | ❌ 需要网络 |
| 成本 | 免费 | 需要付费 |
| 响应速度 | 快（本地） | 取决于网络 |

## 常用命令

```bash
# 查看 Ollama 模型
ollama list

# 删除模型
ollama rm qwen3.5:4b

# 查看 Ollama 日志
journalctl -u ollama -f
```