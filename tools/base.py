from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ToolResult:
    success: bool
    output: str
    error: str | None = None
    metadata: dict = field(default_factory=dict)


class EscalationRequested(Exception):
    def __init__(self, reason: str, refined_prompt: str, context_summary: str):
        self.reason = reason
        self.refined_prompt = refined_prompt
        self.context_summary = context_summary
        super().__init__(reason)


class BaseTool(ABC):
    name: str
    description: str
    parameters_schema: dict
    backends: list[str]  # ["ollama"], ["claude"], or ["ollama", "claude"]
    aliases: list[str] = []  # Alternative names the tool can be looked up by

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult: ...

    def to_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema,
        }
