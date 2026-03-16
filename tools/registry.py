from tools.base import BaseTool


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, BaseTool] = {}
        self._schema_cache: dict[tuple, list[dict]] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool
        for alias in getattr(tool, "aliases", []):
            self._tools[alias] = tool
        self._schema_cache.clear()

    def get_schemas(self, backend: str, enabled_tools: list[str] | None = None) -> list[dict]:
        cache_key = (backend, frozenset(enabled_tools) if enabled_tools is not None else None)
        if cache_key not in self._schema_cache:
            self._schema_cache[cache_key] = [
                tool.to_schema()
                for name, tool in self._tools.items()
                if name == tool.name  # skip alias entries to avoid duplicates
                and backend in tool.backends
                and (enabled_tools is None or tool.name in enabled_tools)
            ]
        return self._schema_cache[cache_key]

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
