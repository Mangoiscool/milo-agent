"""
WebSearchTool 使用示例

演示如何使用网络搜索工具
"""

from core.tools import ToolRegistry, WebSearchTool
from core.tools.builtin import CalculatorTool, WeatherTool


def demo_web_search():
    """演示 WebSearchTool 的基本使用"""
    
    # 方式1: 直接使用 WebSearchTool
    print("=" * 60)
    print("方式1: 直接使用 WebSearchTool")
    print("=" * 60)
    
    # 使用 DuckDuckGo（免费，无需 API Key）
    search_tool = WebSearchTool(engine="duckduckgo")
    
    result = search_tool.execute(query="Python 异步编程", max_results=3)
    print(result.content)
    
    # 方式2: 通过 ToolRegistry 使用
    print("\n" + "=" * 60)
    print("方式2: 通过 ToolRegistry 使用")
    print("=" * 60)
    
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(WeatherTool())
    registry.register(WebSearchTool(engine="duckduckgo"))
    
    # 调用搜索工具
    result = registry.execute("web_search", query="MCP 协议 AI", max_results=3)
    print(result.content)


def demo_tavily_search():
    """演示使用 Tavily 搜索引擎（需要 API Key）"""
    
    print("\n" + "=" * 60)
    print("使用 Tavily 搜索引擎")
    print("=" * 60)
    
    try:
        # 需要 TAVILY_API_KEY 环境变量
        # 或传入 api_key 参数
        search_tool = WebSearchTool(engine="tavily")
        result = search_tool.execute(query="最新 AI 新闻", max_results=5)
        print(result.content)
    except ValueError as e:
        print(f"Tavily 配置错误: {e}")
        print("获取 Tavily API Key: https://tavily.com")


def demo_mixed_tools():
    """演示混合使用多种工具"""
    
    print("\n" + "=" * 60)
    print("混合使用多种工具")
    print("=" * 60)
    
    registry = ToolRegistry()
    
    # 注册本地工具
    registry.register(CalculatorTool())
    registry.register(WeatherTool())
    
    # 注册网络搜索工具
    registry.register(WebSearchTool(engine="duckduckgo"))
    
    print(f"已注册 {registry.count()} 个工具: {registry.list_tools()}")
    
    # 执行不同类型的工具
    print("\n1. 计算器工具:")
    result = registry.execute("calculator", expression="2 ** 10")
    print(f"   结果: {result.content}")
    
    print("\n2. 天气工具:")
    result = registry.execute("weather", city="Beijing", format="simple")
    print(f"   结果: {result.content}")
    
    print("\n3. 网络搜索工具:")
    result = registry.execute("web_search", query="OpenAI GPT-4", max_results=2)
    print(f"   结果: {result.content[:200]}...")


def demo_with_mcp_tools():
    """演示与 MCP 工具混合使用"""
    
    print("\n" + "=" * 60)
    print("与 MCP 工具混合使用")
    print("=" * 60)
    
    registry = ToolRegistry()
    
    # 本地工具
    registry.register(CalculatorTool())
    registry.register(WebSearchTool(engine="duckduckgo"))
    
    # MCP 工具（示例）
    # registry.register_mcp_server("http://localhost:3000/mcp", prefix="mcp")
    
    print(f"工具总数: {registry.count()}")
    print(f"工具列表: {registry.list_tools()}")
    
    # Agent 可以透明地调用任何工具
    # 无论是本地的还是 MCP 的


if __name__ == "__main__":
    print("\n" + "🔍 WebSearchTool 使用示例 " + "🔍".center(50, "="))
    
    # 基本搜索演示
    demo_web_search()
    
    # Tavily 搜索演示（如果有 API Key）
    # demo_tavily_search()
    
    # 混合工具演示
    demo_mixed_tools()
    
    # MCP 集成演示
    demo_with_mcp_tools()
    
    print("\n" + "=" * 60)
    print("提示:")
    print("- DuckDuckGo: 免费，无需 API Key，适合轻量级搜索")
    print("- Tavily: AI 优化，需要 API Key，适合深度搜索")
    print("  获取 API Key: https://tavily.com")
    print("=" * 60)
