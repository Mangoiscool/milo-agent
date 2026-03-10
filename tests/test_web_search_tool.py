"""
测试 WebSearchTool

测试网络搜索工具的基本功能
"""

import pytest
from unittest.mock import Mock, patch

from core.tools import WebSearchTool, ToolRegistry
from core.tools.builtin import WebSearchTool as BuiltinWebSearchTool


class TestWebSearchTool:
    """WebSearchTool 测试类"""
    
    def test_tool_properties(self):
        """测试工具基本属性"""
        tool = WebSearchTool(engine="duckduckgo")
        
        assert tool.name == "web_search"
        assert "搜索" in tool.description
        assert "query" in tool.parameters["properties"]
        assert "max_results" in tool.parameters["properties"]
    
    def test_parameters_schema(self):
        """测试参数 JSON Schema"""
        tool = WebSearchTool(engine="duckduckgo")
        params = tool.parameters
        
        assert params["type"] == "object"
        assert "query" in params["required"]
        assert params["properties"]["max_results"]["minimum"] == 1
        assert params["properties"]["max_results"]["maximum"] == 10
    
    def test_invalid_engine(self):
        """测试无效搜索引擎"""
        with pytest.raises(ValueError) as exc_info:
            WebSearchTool(engine="invalid_engine")
        
        assert "不支持的搜索引擎" in str(exc_info.value)
    
    @patch('core.tools.builtin.web_search.DuckDuckGoEngine.search')
    def test_search_success(self, mock_search):
        """测试成功搜索"""
        # 模拟搜索结果
        mock_search.return_value = [
            {
                "title": "Python 异步编程",
                "url": "https://example.com/async",
                "snippet": "Python 异步编程指南..."
            },
            {
                "title": "asyncio 文档",
                "url": "https://docs.python.org/asyncio",
                "snippet": "Python asyncio 官方文档..."
            }
        ]
        
        tool = WebSearchTool(engine="duckduckgo")
        result = tool.execute(query="Python 异步", max_results=2)
        
        assert not result.is_error
        assert "Python 异步编程" in result.content
        assert "asyncio 文档" in result.content
    
    @patch('core.tools.builtin.web_search.DuckDuckGoEngine.search')
    def test_search_no_results(self, mock_search):
        """测试无结果搜索"""
        mock_search.return_value = []
        
        tool = WebSearchTool(engine="duckduckgo")
        result = tool.execute(query="不存在的查询xyz123", max_results=5)
        
        assert not result.is_error
        assert "未找到相关结果" in result.content
    
    @patch('core.tools.builtin.web_search.DuckDuckGoEngine.search')
    def test_search_error(self, mock_search):
        """测试搜索错误"""
        mock_search.side_effect = Exception("网络错误")
        
        tool = WebSearchTool(engine="duckduckgo")
        result = tool.execute(query="测试", max_results=5)
        
        assert result.is_error
        assert "搜索失败" in result.error_message


class TestWebSearchToolInRegistry:
    """测试 WebSearchTool 在注册中心中的使用"""
    
    def test_register_and_execute(self):
        """测试注册并执行工具"""
        registry = ToolRegistry()
        tool = WebSearchTool(engine="duckduckgo")
        registry.register(tool)
        
        assert registry.has("web_search")
        assert "web_search" in registry.list_tools()
    
    def test_mixed_tools(self):
        """测试混合工具注册"""
        from core.tools.builtin import CalculatorTool
        
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        registry.register(WebSearchTool(engine="duckduckgo"))
        
        tools = registry.list_tools()
        assert "calculator" in tools
        assert "web_search" in tools
    
    def test_get_definitions(self):
        """测试获取工具定义"""
        registry = ToolRegistry()
        registry.register(WebSearchTool(engine="duckduckgo"))
        
        definitions = registry.get_all_definitions()
        assert len(definitions) == 1
        assert definitions[0].name == "web_search"


class TestTavilyEngine:
    """测试 Tavily 搜索引擎"""
    
    def test_missing_api_key(self):
        """测试缺少 API Key"""
        import os
        
        # 清除环境变量
        old_key = os.environ.pop("TAVILY_API_KEY", None)
        
        try:
            with pytest.raises(ValueError) as exc_info:
                from core.tools.builtin.web_search import TavilyEngine
                TavilyEngine()
            
            assert "Tavily API Key 未设置" in str(exc_info.value)
        finally:
            if old_key:
                os.environ["TAVILY_API_KEY"] = old_key


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
