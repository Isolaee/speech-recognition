from datetime import datetime

import httpx

from tools.base import BaseTool, ToolResult
from tools.registry import register_tool


@register_tool(backends=["ollama", "claude"])
class GetCurrentTimeTool(BaseTool):
    name = "get_current_time"
    description = "Return the current local date and time."
    parameters_schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, output=datetime.now().isoformat())


@register_tool(backends=["ollama", "claude"])
class GetWeatherTool(BaseTool):
    name = "get_weather"
    description = "Get current weather for a location given its latitude and longitude."
    parameters_schema = {
        "type": "object",
        "properties": {
            "latitude": {"type": "number", "description": "Latitude of the location"},
            "longitude": {"type": "number", "description": "Longitude of the location"},
        },
        "required": ["latitude", "longitude"],
    }

    def execute(self, latitude: float, longitude: float, **kwargs) -> ToolResult:
        try:
            resp = httpx.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": latitude,
                    "longitude": longitude,
                    "current_weather": True,
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json().get("current_weather", {})
            output = (
                f"Temperature: {data.get('temperature')}°C, "
                f"Wind speed: {data.get('windspeed')} km/h, "
                f"Weather code: {data.get('weathercode')}"
            )
            return ToolResult(success=True, output=output, metadata=data)
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
