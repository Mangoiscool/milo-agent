"""
MCP 工具使用示例

演示如何将 MCP 服务器工具集成到 milo-agent
"""

from core.tools import ToolRegistry, create_mcp_tools_from_server
from core.tools.builtin import CalculatorTool, WeatherTool


def demo_mcp_integration():
    """
    演示 MCP 工具与本地工具的混合使用
    """
    # 1. 创建工具注册中心
    registry = ToolRegistry()
    
    # 2. 注册本地工具
    registry.register(CalculatorTool())
    registry.register(WeatherTool())
    print(f"本地工具: {registry.list_tools()}")
    
    # 3. 连接 MCP 服务器并注册工具（示例）
    # 假设有一个 Open-Meteo 天气 MCP 服务器运行在 localhost:3000
    mcp_server_url = "http://localhost:3000/mcp"
    
    try:
        # 方式一：自动发现并注册所有工具
        count = registry.register_mcp_server(mcp_server_url, prefix="meteo")
        print(f"从 MCP 服务器注册了 {count} 个工具")
        
        # 方式二：手动创建并注册
        # tools = create_mcp_tools_from_server(mcp_server_url)
        # registry.register_mcp_tools(tools)
        
    except Exception as e:
        print(f"连接 MCP 服务器失败（这是正常的，因为是示例）: {e}")
    
    # 4. 获取所有工具定义（传给 LLM）
    definitions = registry.get_all_definitions()
    print(f"\n总共 {len(definitions)} 个工具可用:")
    for defn in definitions:
        print(f"  - {defn.name}: {defn.description[:50]}...")
    
    # 5. 执行工具（本地或 MCP，对 Agent 透明）
    result = registry.execute("calculator", expression="2 + 3 * 4")
    print(f"\n计算结果: {result.content}")
    
    return registry


def demo_weather_mcp():
    """
    演示使用 MCP 天气工具替代 wttr.in
    
    假设有一个 Open-Meteo MCP 服务器
    """
    registry = ToolRegistry()
    
    # 注册 Open-Meteo MCP 工具（国内直连可用）
    # registry.register_mcp_server("http://localhost:3000/mcp", prefix="openmeteo")
    
    # 调用 MCP 天气工具
    # result = registry.execute("openmeteo_get_weather", location="Beijing")
    # print(f"天气: {result.content}")
    
    print("要使用 MCP 天气工具，需要先启动一个 Open-Meteo MCP 服务器")
    print("推荐: https://github.com/anthropics/anthropic-cookbook/tree/main/misc/mcp_server_openmeteo")


if __name__ == "__main__":
    print("=" * 60)
    print("MCP 工具集成示例")
    print("=" * 60)
    
    demo_mcp_integration()
    
    print("\n" + "=" * 60)
    print("天气 MCP 示例")
    print("=" * 60)
    
    demo_weather_mcp()
