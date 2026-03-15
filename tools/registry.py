from tools.base import BaseTool


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get_schemas(self, backend: str) -> list[dict]:
        return [
            tool.to_schema()
            for tool in self._tools.values()
            if backend in tool.backends
        ]

    def lookup(self, name: str) -> BaseTool:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found in registry")
        return self._tools[name]

    def names(self) -> list[str]:
        return list(self._tools.keys())


# Module-level singleton
registry = ToolRegistry()


def register_tool(backends: list[str]):
    """Class decorator: instantiates the class and registers it in the global registry."""
    def decorator(cls):
        instance = cls()
        instance.backends = backends
        registry.register(instance)
        return cls
    return decorator
