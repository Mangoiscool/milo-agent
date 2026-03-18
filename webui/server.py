"""
Milo Agent Web UI Server
FastAPI backend with WebSocket streaming support
"""

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Dict, Optional
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from core.llm.factory import create_llm
from core.llm.base import Message, Role
from agents.main import MainAgent
from core.rag import create_embedding
from core.browser import BrowserConfig


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
    enable_rag: bool = False
    enable_browser: bool = False


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
        self.agents: Dict[str, MainAgent] = {}

    async def create_agent(
        self,
        session_id: str,
        provider: str = "qwen",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        enable_rag: bool = False,
        enable_browser: bool = False
    ) -> MainAgent:
        """为会话创建新 Agent"""
        # 创建 LLM
        if provider == "ollama":
            llm = create_llm("ollama", model=model or "qwen3.5:4b")
        else:
            key = api_key or os.environ.get(f"{provider.upper()}_API_KEY")
            if not key:
                raise ValueError(f"需要设置 {provider.upper()}_API_KEY")
            llm = create_llm(provider, api_key=key, model=model)

        # 创建 embedding（如果启用 RAG）
        embedding_model = None
        if enable_rag:
            try:
                embedding_model = create_embedding("ollama", model="nomic-embed-text")
            except Exception as e:
                print(f"[WARN] Failed to create embedding model: {e}, RAG disabled")
                enable_rag = False

        # 创建 MainAgent
        browser_config = None
        if enable_browser:
            # 非无头模式，用户可以看到浏览器窗口
            browser_config = BrowserConfig(headless=False)

        agent = MainAgent(
            llm=llm,
            enable_builtin_tools=True,
            enable_rag=enable_rag,
            embedding_model=embedding_model,
            enable_browser=enable_browser,
            browser_config=browser_config
        )

        # 注意：浏览器会在首次使用时自动初始化（懒加载）

        self.agents[session_id] = agent
        return agent

    def get_agent(self, session_id: str) -> Optional[MainAgent]:
        """获取会话的 Agent"""
        return self.agents.get(session_id)

    async def remove_agent(self, session_id: str) -> None:
        """移除会话的 Agent"""
        if session_id in self.agents:
            agent = self.agents[session_id]
            # 如果启用了 Browser，需要清理
            if agent.enable_browser:
                await agent.close()
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
        agent = await agent_manager.create_agent(
            session_id=session_id,
            provider=config.provider,
            api_key=config.api_key,
            model=config.model,
            enable_rag=config.enable_rag,
            enable_browser=config.enable_browser
        )
        tool_info = agent.get_tool_info()
        return {
            "success": True,
            "session_id": session_id,
            "tools": tool_info["all_tools"],
            "builtin_tools": tool_info["builtin_tools"],
            "rag_tools": tool_info["rag_tools"],
            "browser_tools": tool_info["browser_tools"],
            "capabilities": {
                "rag": agent.enable_rag,
                "browser": agent.enable_browser
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/agent/{session_id}/tools")
async def get_tools(session_id: str):
    """获取 Agent 可用工具"""
    agent = agent_manager.get_agent(session_id)
    if agent:
        tool_info = agent.get_tool_info()
        return {
            "tools": tool_info["all_tools"],
            "builtin_tools": tool_info["builtin_tools"],
            "rag_tools": tool_info["rag_tools"],
            "browser_tools": tool_info["browser_tools"]
        }
    return {"tools": []}


@app.get("/api/agent/{session_id}/capabilities")
async def get_capabilities(session_id: str):
    """获取 Agent 能力状态"""
    agent = agent_manager.get_agent(session_id)
    if agent:
        return {
            "rag_enabled": agent.enable_rag,
            "browser_enabled": agent.enable_browser,
            "tools_count": len(agent.list_tools())
        }
    return {"rag_enabled": False, "browser_enabled": False, "tools_count": 0}


@app.get("/api/agent/{session_id}/knowledge-base")
async def get_knowledge_base(session_id: str):
    """获取知识库信息"""
    agent = agent_manager.get_agent(session_id)
    if agent and agent.enable_rag:
        return agent.get_knowledge_base_stats()
    return {"enabled": False}


@app.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """WebSocket 聊天接口"""
    await manager.connect(session_id, websocket)

    agent = None
    try:
        # 检查 Agent 是否存在，不存在则创建默认的
        agent = agent_manager.get_agent(session_id)
        if agent is None:
            # 默认使用 qwen，如果没有 API key 则用 ollama
            provider = "ollama" if not os.environ.get("QWEN_API_KEY") else "qwen"
            agent = await agent_manager.create_agent(session_id=session_id, provider=provider)
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
                enable_rag = data.get("enable_rag", False)
                enable_browser = data.get("enable_browser", False)
                try:
                    # 重新创建 Agent 使用新的配置
                    await agent_manager.remove_agent(session_id)
                    agent = await agent_manager.create_agent(
                        session_id=session_id,
                        provider=provider,
                        api_key=api_key,
                        enable_rag=enable_rag,
                        enable_browser=enable_browser
                    )
                    capabilities = []
                    if enable_rag:
                        capabilities.append("RAG")
                    if enable_browser:
                        capabilities.append("Browser")
                    cap_str = f" ({', '.join(capabilities)})" if capabilities else ""
                    await manager.send(session_id, {
                        "type": "info",
                        "message": f"已切换到 {provider} 模型{cap_str}",
                        "provider": provider,
                        "capabilities": {
                            "rag": enable_rag,
                            "browser": enable_browser
                        }
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

                            result = await agent.tool_registry.aexecute(tool_call.name, **tool_call.arguments)

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
        # 清理资源
        if agent and agent.enable_browser:
            await agent.close()
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
