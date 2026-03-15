from dataclasses import dataclass, field


@dataclass
class Skill:
    name: str
    description: str
    system_prompt: str
    enabled_tools: list[str]
    tts_voice: str | None = None
    ollama_model_override: str | None = None
    always_escalate: bool = False
    escalation_threshold: float = 0.8
