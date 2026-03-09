"""
天气查询工具

功能：通过 wttr.in 查询天气（免费，无需 API Key）
"""

import json
import os
from typing import Any, Dict

import httpx

from ..base import BaseTool, ToolResult


class WeatherTool(BaseTool):
    """
    天气查询工具
    
    使用 wttr.in 免费 API
    无需 API Key
    
    功能：
    - 查询城市天气
    - 支持多种格式输出
    - 自动检测代理配置
    """
    
    WTTR_URL = "https://wttr.in"
    
    def _get_proxy(self) -> str:
        """获取代理配置"""
        # 从环境变量读取代理
        https_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
        if https_proxy:
            return https_proxy
        return None
    
    @property
    def name(self) -> str:
        return "weather"
    
    @property
    def description(self) -> str:
        return """查询指定城市的天气信息。

功能：
- 获取城市当前天气
- 包含温度、湿度、风速、天气状况等

参数：
- city: 城市名称（如 "北京", "Beijing", "Shanghai"）
- format: 输出格式
  - simple: 简洁格式（默认）
  - detailed: 详细格式
  - json: JSON 格式

示例：
- city="北京" → "北京: ☀️ 晴, 18°C, 湿度 45%"
- city="Shanghai", format="detailed" → 详细天气信息
- city="Tokyo", format="json" → JSON 格式数据

注意：使用 wttr.in 免费服务，无需 API Key。"""

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称（中文或英文）"
                },
                "format": {
                    "type": "string",
                    "enum": ["simple", "detailed", "json"],
                    "description": "输出格式：simple（简洁）, detailed（详细）, json（JSON）"
                }
            },
            "required": ["city"]
        }
    
    def execute(self, city: str, format: str = "simple", **kwargs) -> ToolResult:
        """
        查询天气
        
        Args:
            city: 城市名称
            format: 输出格式
        
        Returns:
            天气信息
        """
        try:
            if format == "json":
                return self._get_json(city)
            elif format == "detailed":
                return self._get_detailed(city)
            else:
                return self._get_simple(city)
                
        except httpx.HTTPError as e:
            return ToolResult(
                content="",
                is_error=True,
                error_message=f"网络请求失败: {str(e)}"
            )
        except Exception as e:
            return ToolResult(
                content="",
                is_error=True,
                error_message=f"查询天气失败: {str(e)}"
            )
    
    def _get_simple(self, city: str) -> ToolResult:
        """获取简洁格式天气"""
        url = f"{self.WTTR_URL}/{city}?format=%l:+%c+%t,+%h"
        
        proxy = self._get_proxy()
        with httpx.Client(timeout=10.0, proxy=proxy) as client:
            response = client.get(url)
            response.raise_for_status()
            result = response.text.strip()
        
        return ToolResult(content=result)
    
    def _get_detailed(self, city: str) -> ToolResult:
        """获取详细格式天气"""
        url = f"{self.WTTR_URL}/{city}?lang=zh"
        
        proxy = self._get_proxy()
        with httpx.Client(timeout=10.0, proxy=proxy) as client:
            response = client.get(url)
            response.raise_for_status()
            result = response.text.strip()
        
        # 只返回前几行（当前天气）
        lines = result.split('\n')[:7]
        return ToolResult(content='\n'.join(lines))
    
    def _get_json(self, city: str) -> ToolResult:
        """获取 JSON 格式天气"""
        url = f"{self.WTTR_URL}/{city}?format=j1"
        
        proxy = self._get_proxy()
        with httpx.Client(timeout=10.0, proxy=proxy) as client:
            response = client.get(url)
            response.raise_for_status()
            data = response.json()
        
        # 提取关键信息
        current = data.get("current_condition", [{}])[0]
        result = {
            "city": city,
            "temperature": f"{current.get('temp_C')}°C",
            "feels_like": f"{current.get('FeelsLikeC')}°C",
            "description": current.get("weatherDesc", [{}])[0].get("value", ""),
            "humidity": f"{current.get('humidity')}%",
            "wind": f"{current.get('windspeedKmph')} km/h",
            "wind_direction": current.get("winddir16Point"),
        }
        
        return ToolResult(content=json.dumps(result, ensure_ascii=False, indent=2))


__all__ = ["WeatherTool"]
