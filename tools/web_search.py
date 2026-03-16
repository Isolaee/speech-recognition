from ddgs import DDGS

from tools.base import BaseTool, ToolResult
from tools.registry import register_tool


@register_tool(backends=["ollama", "claude"])
class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Search the web using DuckDuckGo and return a list of results."
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (default 5)",
                "default": 5,
            },
        },
        "required": ["query"],
    }

    def execute(self, query: str, max_results: int = 5, **kwargs) -> ToolResult:
        try:
            results = DDGS().text(query, max_results=max_results)
            if not results:
                return ToolResult(success=True, output="No results found.")
            lines = [
                f"{i+1}. {r['title']}\n   {r['href']}\n   {r['body']}"
                for i, r in enumerate(results)
            ]
            return ToolResult(success=True, output="\n\n".join(lines))
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
