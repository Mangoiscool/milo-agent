"""
Milo Agent Web UI Server
FastAPI backend with WebSocket streaming support
"""

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Dict, List, Optional
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from core.llm.factory import create_llm
from core.llm.base import Message, Role
from agents.milo_agent import MiloAgent
from agents.base import AgentEvent
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
        self.agents: Dict[str, MiloAgent] = {}

    async def create_agent(
        self,
        session_id: str,
        provider: str = "qwen",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        enable_rag: bool = False,
        enable_browser: bool = False
    ) -> MiloAgent:
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

        # 创建 MiloAgent
        browser_config = None
        if enable_browser:
            # 非无头模式，用户可以看到浏览器窗口
            browser_config = BrowserConfig(headless=False)

        agent = MiloAgent(
            llm=llm,
            enable_builtin_tools=True,
            embedding_model=embedding_model,
            enable_browser=enable_browser,
            browser_config=browser_config
        )

        # 注意：浏览器会在首次使用时自动初始化（懒加载）

        self.agents[session_id] = agent
        return agent

    def get_agent(self, session_id: str) -> Optional[MiloAgent]:
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
# Chat Event Handler
# ═══════════════════════════════════════════════════════════════

async def handle_chat_with_events(
    agent: MiloAgent,
    message: str,
    session_id: str
) -> None:
    """
    处理对话并发送事件

    使用 MiloAgent 的 chat() 方法，通过事件系统捕获工具调用
    """
    collected_events: List[dict] = []

    def on_tool_call(name: str, arguments: dict):
        collected_events.append({
            "type": "tool_call",
            "name": name,
            "args": str(arguments)
        })

    def on_tool_result(name: str, result: str, is_error: bool):
        truncated = result[:500] if result else ""
        collected_events.append({
            "type": "tool_result",
            "name": name,
            "result": truncated,
            "is_error": is_error
        })

    agent.on(AgentEvent.TOOL_CALL, on_tool_call)
    agent.on(AgentEvent.TOOL_RESULT, on_tool_result)

    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: agent.chat(message)
        )

        for event in collected_events:
            await manager.send(session_id, event)

        await manager.send(session_id, {
            "type": "assistant",
            "content": response
        })

    finally:
        agent.off(AgentEvent.TOOL_CALL, on_tool_call)
        agent.off(AgentEvent.TOOL_RESULT, on_tool_result)


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
            {"id": "qwen", "name": "通义千问", "needs_key": True, "default_model": "MiniMax-M2.1"},
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

            # 使用统一的事件处理
            await handle_chat_with_events(agent, message, session_id)

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
