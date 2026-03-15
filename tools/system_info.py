import platform

import psutil

from tools.base import BaseTool, ToolResult
from tools.registry import register_tool


@register_tool(backends=["ollama", "claude"])
class SystemInfoTool(BaseTool):
    name = "get_system_info"
    description = "Return current system resource usage: CPU, RAM, battery, and OS."
    parameters_schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def execute(self, **kwargs) -> ToolResult:
        try:
            cpu = psutil.cpu_percent(interval=0.5)
            ram = psutil.virtual_memory()
            battery = psutil.sensors_battery()

            lines = [
                f"OS: {platform.system()} {platform.release()}",
                f"CPU: {cpu}%",
                f"RAM: {ram.percent}% used ({ram.used // 1024 // 1024} MB / {ram.total // 1024 // 1024} MB)",
            ]
            if battery:
                lines.append(
                    f"Battery: {battery.percent:.0f}% "
                    f"({'charging' if battery.power_plugged else 'discharging'})"
                )

            return ToolResult(success=True, output="\n".join(lines))
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
