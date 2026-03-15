from __future__ import annotations

from typing import TYPE_CHECKING

from config import EscalationConfig
from llm.base import LLMResponse
from tools.base import EscalationRequested
from tools.registry import ToolRegistry
from utils.logger import get_logger

if TYPE_CHECKING:
    from llm.claude_backend import ClaudeBackend
    from llm.ollama_backend import OllamaBackend
    from skills.base import Skill
    from voice.tts import PiperBackend

logger = get_logger(__name__)


class LLMRouter:
    def __init__(
        self,
        ollama: OllamaBackend,
        claude: ClaudeBackend,
        registry: ToolRegistry,
        config: EscalationConfig,
    ):
        self.ollama = ollama
        self.claude = claude
        self.registry = registry
        self.config = config

    def chat(
        self,
        messages: list[dict],
        skill: Skill | None,
        tts: PiperBackend | None = None,
        system: str = "",
    ) -> LLMResponse:
        enabled_tools = skill.enabled_tools if skill else None
        always_escalate = skill.always_escalate if skill else False
        model_override = skill.ollama_model_override if skill else None
        tts_voice = skill.tts_voice if skill else None

        # Direct escalation to Claude if the skill demands it
        if always_escalate:
            logger.info("Skill '%s' has always_escalate=True; routing to Claude.", skill.name if skill else "unknown")
            claude_tools = self.registry.get_schemas("claude", enabled_tools)
            return self.claude.chat(messages, claude_tools, system)

        # Try Ollama first
        ollama_tools = self.registry.get_schemas("ollama", enabled_tools)
        try:
            return self.ollama.chat(messages, ollama_tools, system, model_override=model_override)
        except EscalationRequested as exc:
            logger.info("Escalating to Claude: %s", exc.reason)
            if tts:
                tts.speak("Let me use a more powerful model for this one.", voice_override=tts_voice)

            escalation_messages = [
                {"role": "user", "content": exc.refined_prompt},
                {"role": "user", "content": f"[Context summary: {exc.context_summary}]"},
            ]
            claude_tools = self.registry.get_schemas("claude", enabled_tools)
            return self.claude.chat(escalation_messages, claude_tools, system)
