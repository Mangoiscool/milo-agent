"""
网络搜索工具

功能：支持多种搜索引擎（Tavily、DuckDuckGo）
"""

import json
import os
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod

import httpx

from ..base import BaseTool, ToolResult


# ═══════════════════════════════════════════════════════════════
# 搜索引擎抽象
# ═══════════════════════════════════════════════════════════════

class SearchEngine(ABC):
    """搜索引擎抽象基类"""
    
    @abstractmethod
    def search(self, query: str, max_results: int) -> List[Dict[str, str]]:
        """
        执行搜索
        
        Args:
            query: 搜索关键词
            max_results: 最大结果数
        
        Returns:
            搜索结果列表，每项包含 title, url, snippet
        """
        pass


class TavilyEngine(SearchEngine):
    """
    Tavily 搜索引擎
    
    特点：
    - AI 优化的搜索结果
    - 专为 LLM 设计
    - 需要API Key（https://tavily.com）
    """
    
    API_URL = "https://api.tavily.com/search"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Tavily API Key 未设置。请设置环境变量 TAVILY_API_KEY "
                "或在初始化时传入 api_key 参数。获取 API Key: https://tavily.com"
            )
    
    def search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """执行 Tavily 搜索"""
        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",  # basic 或 advanced
        }
        
        with httpx.Client(timeout=30.0) as client:
            response = client.post(self.API_URL, json=payload)
            response.raise_for_status()
            data = response.json()
        
        results = []
        for item in data.get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", ""),
            })
        
        return results


class DuckDuckGoEngine(SearchEngine):
    """
    DuckDuckGo 搜索引擎
    
    特点：
    - 完全免费，无需 API Key
    - 隐私友好
    - 适合轻量级搜索
    """
    
    API_URL = "https://api.duckduckgo.com/"
    
    def __init__(self):
        # 尝试使用 duckduckgo-search 库（更可靠）
        self._use_library = False
        try:
            from duckduckgo_search import DDGS
            self._ddgs = DDGS()
            self._use_library = True
        except ImportError:
            pass  # 使用 HTTP API
    
    def search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """执行 DuckDuckGo 搜索"""
        if self._use_library:
            return self._search_with_library(query, max_results)
        else:
            return self._search_with_api(query, max_results)
    
    def _search_with_library(self, query: str, max_results: int) -> List[Dict[str, str]]:
        """使用 duckduckgo-search 库"""
        from duckduckgo_search import DDGS
        
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                })
        
        return results
    
    def _search_with_api(self, query: str, max_results: int) -> List[Dict[str, str]]:
        """使用 DuckDuckGo HTTP API（功能有限）"""
        params = {
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1,
        }
        
        with httpx.Client(timeout=10.0) as client:
            response = client.get(self.API_URL, params=params)
            response.raise_for_status()
            data = response.json()
        
        results = []
        
        # DuckDuckGo API 的 Instant Answer
        if data.get("AbstractText"):
            results.append({
                "title": data.get("Heading", "摘要"),
                "url": data.get("AbstractURL", ""),
                "snippet": data.get("AbstractText", ""),
            })
        
        # Related Topics
        for topic in data.get("RelatedTopics", [])[:max_results - len(results)]:
            if isinstance(topic, dict) and "Text" in topic:
                results.append({
                    "title": topic.get("FirstURL", "").split("/")[-1] or "相关主题",
                    "url": topic.get("FirstURL", ""),
                    "snippet": topic.get("Text", ""),
                })
        
        return results[:max_results]


# ═══════════════════════════════════════════════════════════════
# WebSearchTool 实现
# ═══════════════════════════════════════════════════════════════

class WebSearchTool(BaseTool):
    """
    网络搜索工具
    
    支持多种搜索引擎：
    - tavily: AI 优化搜索（推荐，需要 API Key）
    - duckduckgo: 免费搜索（无需 API Key）
    
    环境变量：
    - TAVILY_API_KEY: Tavily API Key（使用 Tavily 时需要）
    - SEARCH_ENGINE: 默认搜索引擎（tavily 或 duckduckgo）
    """
    
    def __init__(
        self,
        engine: str = "duckduckgo",
        api_key: Optional[str] = None
    ):
        """
        初始化搜索工具
        
        Args:
            engine: 搜索引擎（tavily 或 duckduckgo）
            api_key: API Key（仅 Tavily 需要）
        """
        super().__init__()
        
        # 从环境变量获取默认引擎
        self.engine_name = engine or os.environ.get("SEARCH_ENGINE", "duckduckgo")
        
        # 初始化搜索引擎
        if self.engine_name == "tavily":
            self._engine = TavilyEngine(api_key)
        elif self.engine_name == "duckduckgo":
            self._engine = DuckDuckGoEngine()
        else:
            raise ValueError(f"不支持的搜索引擎: {self.engine_name}。支持: tavily, duckduckgo")
    
    @property
    def name(self) -> str:
        return "web_search"
    
    @property
    def description(self) -> str:
        return f"""搜索互联网获取实时信息。

功能：
- 搜索最新新闻、技术文档、教程等
- 返回相关网页的标题、链接和摘要
- 当前使用引擎: {self.engine_name}

参数：
- query: 搜索关键词或问题
- max_results: 返回结果数量（1-10，默认5）

示例：
- query="Python 异步编程最佳实践" → 返回相关文章列表
- query="2024年AI发展趋势", max_results=3 → 返回3条结果

注意：
- {self._get_engine_notice()}
"""
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词或问题"
                },
                "max_results": {
                    "type": "integer",
                    "description": "返回结果数量（1-10）",
                    "minimum": 1,
                    "maximum": 10,
                }
            },
            "required": ["query"]
        }
    
    def _get_engine_notice(self) -> str:
        """获取引擎说明"""
        if self.engine_name == "tavily":
            return "使用 Tavily API，需要设置 TAVILY_API_KEY 环境变量"
        else:
            return "使用 DuckDuckGo 免费 API，无需 API Key"
    
    def execute(self, query: str, max_results: int = 5, **kwargs) -> ToolResult:
        """
        执行搜索
        
        Args:
            query: 搜索关键词
            max_results: 最大结果数
        
        Returns:
            搜索结果
        """
        try:
            self.logger.info(f"Searching with {self.engine_name}: {query}")
            
            results = self._engine.search(query, max_results)
            
            if not results:
                return ToolResult(
                    content="未找到相关结果",
                    is_error=False
                )
            
            # 格式化输出
            output_lines = [f"搜索结果（{self.engine_name}）:\n"]
            
            for i, result in enumerate(results, 1):
                output_lines.append(f"{i}. {result['title']}")
                output_lines.append(f"   链接: {result['url']}")
                output_lines.append(f"   摘要: {result['snippet'][:200]}...")
                output_lines.append("")
            
            content = "\n".join(output_lines)
            
            return ToolResult(content=content)
            
        except httpx.HTTPError as e:
            return ToolResult(
                content="",
                is_error=True,
                error_message=f"网络请求失败: {str(e)}"
            )
        except ValueError as e:
            return ToolResult(
                content="",
                is_error=True,
                error_message=str(e)
            )
        except Exception as e:
            return ToolResult(
                content="",
                is_error=True,
                error_message=f"搜索失败: {str(e)}"
            )


__all__ = ["WebSearchTool", "TavilyEngine", "DuckDuckGoEngine"]
