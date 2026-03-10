# 高级示例 - 完整 Agent 演示

这个示例展示了 Milo Agent 的完整功能，包括工具调用和对话能力。

## 功能特性

- 工具注册和管理
- 支持多种 LLM 提供者
- 工具调用（Function Calling）
- 交互式对话
- 批量测试模式

## 使用方法

### 1. 交互模式

```bash
# 使用默认的 Qwen 提供者（需要设置 API key）
export QWEN_API_KEY='your-api-key'
python examples/advanced/complete_agent.py

# 指定其他 LLM 提供者
export GLM_API_KEY='your-api-key'
python examples/advanced/complete_agent.py --provider glm
```

### 2. 批量测试模式

```bash
# 使用 Qwen 测试
python examples/advanced/complete_agent.py --provider qwen "你好" "帮我计算 2+2"

# 使用 GLM 测试
python examples/advanced/complete_agent.py --provider glm "介绍一下你自己"

# 使用 Ollama（本地，不支持工具调用）
python examples/advanced/complete_agent.py --provider ollama "你好"
```

### 3. 支持的 LLM 提供者

- **qwen**: 通义千问（需要 QWEN_API_KEY）
- **glm**: 智谱 GLM（需要 GLM_API_KEY）
- **deepseek**: DeepSeek（需要 DEEPSEEK_API_KEY）
- **ollama**: 本地模型（需要安装 Ollama）

## 环境配置

### Ollama 用户

1. 安装 Ollama：https://ollama.ai
2. 下载模型：
   ```bash
   ollama pull qwen3.5:4b
   ```
3. 启动服务：
   ```bash
   ollama serve
   ```

### API 用户

设置对应的环境变量：
```bash
export QWEN_API_KEY='your-api-key'
# 或
export GLM_API_KEY='your-api-key'
# 或
export DEEPSEEK_API_KEY='your-api-key'
```

## 可用工具

- **calculator**: 计算器工具
- **weather**: 天气查询工具
- **web_search**: 网络搜索工具
- **file_read**: 文件读取工具
- **code_execute**: 代码执行工具

## 注意事项

1. **API 提供者**: 只有 API 提供者（qwen, glm, deepseek）支持工具调用
2. **Ollama**: Ollama 不支持工具调用，只能进行普通对话
3. **网络连接**: 使用网络工具需要网络连接
4. **文件权限**: 文件操作需要相应的文件系统权限