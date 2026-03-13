"""
Milo Agent Web UI Server
FastAPI backend with WebSocket streaming support
"""

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Dict, Optional, Set
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core.llm.factory import create_llm
from core.llm.base import Message, Role, ToolCall
from agents.simple import SimpleAgent, AgentConfig, AgentEvent
from core.tools import ToolRegistry
from core.tools.builtin import (
    CalculatorTool,
    WeatherTool,
    WebSearchTool,
    FileReadTool,
    CodeExecutionTool
)


# ═══════════════════════════════════════════════════════════════
# Request/Response Models
# ═══════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    message: str
    session_id: str


class ProviderConfig(BaseModel):
    provider: str = "qwen"
    api_key: Optional[str] = None
    model: Optional[str] = None


# ═══════════════════════════════════════════════════════════════
# Connection Manager
# ═══════════════════════════════════════════════════════════════

class ConnectionManager:
    """管理 WebSocket 连接"""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, session_id: str, websocket: WebSocket) -> None:
        """接受新连接"""
        await websocket.accept()
        self.active_connections[session_id] = websocket

    def disconnect(self, session_id: str) -> None:
        """断开连接"""
        if session_id in self.active_connections:
            del self.active_connections[session_id]

    async def send(self, session_id: str, data: dict) -> bool:
        """发送消息到指定会话"""
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_json(data)
                return True
            except Exception:
                self.disconnect(session_id)
        return False

    async def broadcast(self, data: dict) -> None:
        """广播消息到所有连接"""
        for session_id, connection in list(self.active_connections.items()):
            await self.send(session_id, data)


manager = ConnectionManager()


# ═══════════════════════════════════════════════════════════════
# Agent Manager
# ═══════════════════════════════════════════════════════════════

class AgentManager:
    """管理 Agent 实例"""

    def __init__(self):
        self.agents: Dict[str, SimpleAgent] = {}

    def create_agent(self, session_id: str, provider: str = "qwen", api_key: Optional[str] = None, model: Optional[str] = None) -> SimpleAgent:
        """为会话创建新 Agent"""
        if provider == "ollama":
            llm = create_llm("ollama", model=model or "qwen3.5:4b")
            # Ollama 不支持工具调用
            agent = SimpleAgent(llm, config=AgentConfig(system_prompt="你是一个有用的助手。"))
        else:
            # API 提供者
            key = api_key or os.environ.get(f"{provider.upper()}_API_KEY")
            if not key:
                raise ValueError(f"需要设置 {provider.upper()}_API_KEY")

            llm = create_llm(provider, api_key=key, model=model)

            # 注册工具
            registry = ToolRegistry()
            registry.register(CalculatorTool())
            registry.register(WeatherTool())
            registry.register(WebSearchTool(engine="duckduckgo"))
            registry.register(FileReadTool())
            registry.register(CodeExecutionTool())

            # 简化的系统提示词
            system_prompt = """你是一个有用的 AI 助手。你必须使用可用的工具来回答用户的问题：

- 计算 → calculator 工具
- 天气查询 → weather 工具
- 网络搜索 → web_search 工具
- 读取文件 → file_read 工具
- 执行代码 → code_execute 工具

重要：
1. 当用户需要计算、查询天气、搜索等时，必须调用相应工具
2. 不要自己计算或猜测，使用工具获取准确结果
3. 直接回答用户的问题，不要介绍自己"""

            agent = SimpleAgent(
                llm,
                tools=list(registry._tools.values()),
                config=AgentConfig(system_prompt=system_prompt)
            )

        self.agents[session_id] = agent
        return agent

    def get_agent(self, session_id: str) -> Optional[SimpleAgent]:
        """获取会话的 Agent"""
        return self.agents.get(session_id)

    def remove_agent(self, session_id: str) -> None:
        """移除会话的 Agent"""
        if session_id in self.agents:
            del self.agents[session_id]


agent_manager = AgentManager()


# ═══════════════════════════════════════════════════════════════
# FastAPI App
# ═══════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    print("\n" + "=" * 60)
    print("  Milo Agent Web UI Server")
    print("=" * 60)
    print("  访问 http://localhost:8000")
    print("=" * 60 + "\n")
    yield
    print("\nServer shutdown")


app = FastAPI(lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def get_index():
    """返回主页"""
    html_file = os.path.join(os.path.dirname(__file__), "static", "index.html")
    with open(html_file, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/providers")
async def get_providers():
    """获取支持的 LLM 提供者"""
    return {
        "providers": [
            {"id": "qwen", "name": "通义千问", "needs_key": True, "default_model": "qwen-plus"},
            {"id": "glm", "name": "智谱 GLM", "needs_key": True, "default_model": "glm-4-flash"},
            {"id": "deepseek", "name": "DeepSeek", "needs_key": True, "default_model": "deepseek-chat"},
            {"id": "ollama", "name": "Ollama (本地)", "needs_key": False, "default_model": "qwen3.5:4b"},
        ]
    }


@app.post("/api/agent/create")
async def create_agent(config: ProviderConfig):
    """创建新 Agent 会话"""
    session_id = str(uuid4())
    try:
        agent = agent_manager.create_agent(
            session_id=session_id,
            provider=config.provider,
            api_key=config.api_key,
            model=config.model
        )
        return {
            "success": True,
            "session_id": session_id,
            "tools": agent.list_tools()
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/agent/{session_id}/tools")
async def get_tools(session_id: str):
    """获取 Agent 可用工具"""
    agent = agent_manager.get_agent(session_id)
    if agent:
        return {"tools": agent.list_tools()}
    return {"tools": []}


@app.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """WebSocket 聊天接口"""
    await manager.connect(session_id, websocket)

    try:
        # 检查 Agent 是否存在，不存在则创建默认的
        agent = agent_manager.get_agent(session_id)
        if agent is None:
            # 默认使用 qwen，如果没有 API key 则用 ollama
            provider = "ollama" if not os.environ.get("QWEN_API_KEY") else "qwen"
            agent = agent_manager.create_agent(session_id=session_id, provider=provider)
            await manager.send(session_id, {
                "type": "info",
                "message": f"使用 {provider} 模型",
                "provider": provider
            })

        while True:
            data = await websocket.receive_json()

            # 处理配置消息
            if data.get("type") == "config":
                provider = data.get("provider", "ollama")
                api_key = data.get("api_key")
                try:
                    # 重新创建 Agent 使用新的配置
                    agent_manager.remove_agent(session_id)
                    agent = agent_manager.create_agent(
                        session_id=session_id,
                        provider=provider,
                        api_key=api_key
                    )
                    await manager.send(session_id, {
                        "type": "info",
                        "message": f"已切换到 {provider} 模型",
                        "provider": provider
                    })
                    await manager.send(session_id, {"type": "config_ack"})
                except Exception as e:
                    await manager.send(session_id, {
                        "type": "error",
                        "message": f"切换模型失败: {str(e)}"
                    })
                continue

            message = data.get("message", "")
            if not message:
                continue

            if not message:
                continue

            # 发送用户消息确认
            await manager.send(session_id, {"type": "user", "content": message})

            # 添加用户消息到 memory
            agent.memory.add(Message(role=Role.USER, content=message))

            # 处理对话
            if agent.tool_registry.count() > 0 and hasattr(agent, 'chat_with_tools'):
                # 有工具支持
                for iteration in range(10):
                    messages = agent._build_messages()

                    # 调试日志
                    print(f"[DEBUG] Iteration {iteration}, messages: {len(messages)}")
                    print(f"[DEBUG] Tools available: {[t.name for t in agent.tool_registry.get_all_definitions()]}")

                    # 打印系统提示词
                    for msg in messages:
                        if msg.role.value == "system":
                            print(f"[DEBUG] System prompt (first 200 chars): {msg.content[:200]}...")

                    # 打印用户消息
                    for msg in messages:
                        if msg.role.value == "user":
                            print(f"[DEBUG] User message: {msg.content}")

                    response = agent.llm.chat_with_tools(
                        messages,
                        tools=agent.tool_registry.get_all_definitions()
                    )

                    # 调试日志
                    print(f"[DEBUG] Response: finish_reason={response.finish_reason}")
                    print(f"[DEBUG] Response content: {response.content[:100] if response.content else 'empty'}")
                    print(f"[DEBUG] Tool calls: {len(response.tool_calls) if response.tool_calls else 0}")
                    if response.tool_calls:
                        for tc in response.tool_calls:
                            print(f"[DEBUG]   - {tc.name}: {tc.arguments}")

                    # 检查是否有工具调用（只要有 tool_calls 就执行）
                    if response.tool_calls:
                        # 保存 assistant 消息（包含 tool_calls）
                        agent.memory.add(Message(
                            role=Role.ASSISTANT,
                            content=response.content or "",
                            tool_calls=response.tool_calls
                        ))

                        # 如果有文本内容，发送回复
                        if response.content:
                            await manager.send(session_id, {
                                "type": "assistant",
                                "content": response.content
                            })

                        # 执行工具调用
                        for tool_call in response.tool_calls:
                            await manager.send(session_id, {
                                "type": "tool_call",
                                "name": tool_call.name,
                                "args": str(tool_call.arguments)
                            })

                            result = agent.tool_registry.execute(tool_call.name, **tool_call.arguments)

                            await manager.send(session_id, {
                                "type": "tool_result",
                                "name": tool_call.name,
                                "result": result.content[:500] if result.content else "",
                                "is_error": result.is_error
                            })

                            # 保存工具结果
                            agent.memory.add(Message(
                                role=Role.TOOL,
                                content=result.content if not result.is_error else f"Error: {result.error_message}",
                                name=tool_call.name,
                                tool_call_id=tool_call.id
                            ))
                    else:
                        # 不需要调用工具，发送最终回复并退出
                        agent.memory.add(Message(role=Role.ASSISTANT, content=response.content))
                        await manager.send(session_id, {
                            "type": "assistant",
                            "content": response.content
                        })
                        break
            else:
                # 普通对话
                response = agent.chat(message)
                await manager.send(session_id, {"type": "assistant", "content": response})

    except WebSocketDisconnect:
        print(f"Client disconnected: {session_id}")
    except Exception as e:
        await manager.send(session_id, {"type": "error", "message": str(e)})
    finally:
        manager.disconnect(session_id)


def main():
    """启动服务器"""
    uvicorn.run(
        "webui.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )


if __name__ == "__main__":
    main()
