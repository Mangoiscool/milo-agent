"""
基于搜索的天气查询工具

功能：通过搜索引擎查询天气（替代直接调用 wttr.in）
"""

import re
from typing import Any, Dict

from ..base import BaseTool, ToolResult
from .web_search import WebSearchTool


class WeatherSearchTool(BaseTool):
    """
    基于搜索的天气查询工具
    
    使用 WebSearchTool 搜索天气信息，而非直接调用天气 API
    
    优势：
    - 不依赖特定天气服务
    - 避免网络限制（如 wttr.in 国内访问问题）
    - 返回多源信息
    """
    
    def __init__(self, search_engine: str = "duckduckgo"):
        """
        初始化天气搜索工具
        
        Args:
            search_engine: 搜索引擎（duckduckgo 或 tavily）
        """
        super().__init__()
        self._search_tool = WebSearchTool(engine=search_engine)
    
    @property
    def name(self) -> str:
        return "weather_search"
    
    @property
    def description(self) -> str:
        return """通过搜索引擎查询城市天气信息。

功能：
- 搜索指定城市的实时天气
- 返回温度、天气状况、风向等信息
- 信息来源多样化（不依赖单一 API）

参数：
- city: 城市名称（如 "北京", "Beijing", "Shanghai"）
- lang: 语言（zh 或 en，默认 zh）

示例：
- city="北京" → "北京天气: 晴, 18°C, 西北风 3级"
- city="Tokyo", lang="en" → "Tokyo weather: Sunny, 22°C"

注意：使用搜索引擎获取天气，信息可能略有延迟。"""
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称（中文或英文）"
                },
                "lang": {
                    "type": "string",
                    "enum": ["zh", "en"],
                    "description": "语言（zh 中文, en 英文）"
                }
            },
            "required": ["city"]
        }
    
    def execute(self, city: str, lang: str = "zh", **kwargs) -> ToolResult:
        """
        搜索天气信息
        
        Args:
            city: 城市名称
            lang: 语言
        
        Returns:
            天气信息
        """
        try:
            # 构建搜索查询
            if lang == "zh":
                query = f"{city}天气 实时"
            else:
                query = f"{city} weather today"
            
            self.logger.info(f"Searching weather for: {city}")
            
            # 使用 WebSearchTool 搜索
            search_result = self._search_tool.execute(query=query, max_results=3)
            
            if search_result.is_error:
                return search_result
            
            # 提取天气信息
            weather_info = self._extract_weather_info(search_result.content, city, lang)
            
            return ToolResult(content=weather_info)
            
        except Exception as e:
            return ToolResult(
                content="",
                is_error=True,
                error_message=f"天气查询失败: {str(e)}"
            )
    
    def _extract_weather_info(self, search_content: str, city: str, lang: str) -> str:
        """
        从搜索结果中提取天气信息
        
        Args:
            search_content: 搜索结果内容
            city: 城市名称
            lang: 语言
        
        Returns:
            提取的天气信息
        """
        lines = search_content.split('\n')
        
        # 提取关键信息
        weather_lines = []
        for line in lines:
            # 跳过空行和标题行
            if not line.strip() or '搜索结果' in line:
                continue
            
            # 查找包含天气相关信息的行
            if any(keyword in line for keyword in 
                   ['°C', '°F', '温度', '天气', '晴', '雨', 'cloud', 'sunny', 'rain', 'temp', '°']):
                # 清理格式
                clean_line = line.strip()
                if clean_line.startswith(('1.', '2.', '3.')):
                    clean_line = clean_line[2:].strip()
                weather_lines.append(clean_line)
        
        if not weather_lines:
            return f"{city}天气信息未找到，建议直接查看天气网站"
        
        # 返回前 3 条相关信息
        result_lines = [f"{city}天气信息（基于搜索）:\n"]
        result_lines.extend(weather_lines[:3])
        
        return '\n'.join(result_lines)


__all__ = ["WeatherSearchTool"]
