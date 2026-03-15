from __future__ import annotations

from typing import TYPE_CHECKING

from llm.base import ToolCall

if TYPE_CHECKING:
    from skills.base import Skill

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Be concise and clear in your responses."
)


class ConversationContext:
    def __init__(self, max_turns: int, skill: "Skill | None" = None):
        self.max_turns = max_turns
        self._skill: "Skill | None" = skill
        self._history: list[dict] = []

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def add_user(self, text: str) -> None:
        self._history.append({"role": "user", "content": text})

    def add_assistant(self, text: str) -> None:
        self._history.append({"role": "assistant", "content": text})

    def add_tool_call(self, tool_call: ToolCall) -> None:
        """Append an assistant message representing a tool call (Anthropic format)."""
        self._history.append({
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": tool_call.id,
                    "name": tool_call.name,
                    "input": tool_call.arguments,
                }
            ],
        })

    def add_tool_result(self, tool_id: str, result: str) -> None:
        """Append a tool result message (Anthropic format)."""
        self._history.append({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": result,
                }
            ],
        })

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def to_messages(self) -> list[dict]:
        """Return history applying a sliding window by turn count.

        A "turn" is one user+assistant exchange. We keep the last
        ``max_turns`` complete exchanges.
        """
        # Collect full turn pairs (user → assistant) respecting max_turns.
        # Walk backwards and count turns.
        turns: list[list[dict]] = []
        current_turn: list[dict] = []

        for msg in reversed(self._history):
            current_turn.insert(0, msg)
            if msg["role"] == "user" and not isinstance(msg.get("content"), list):
                # A plain user message marks the start of a turn.
                turns.insert(0, current_turn)
                current_turn = []
                if len(turns) >= self.max_turns:
                    break

        # Flatten the retained turns
        trimmed = [msg for turn in turns for msg in turn]

        # If there were trailing tool-call messages not yet part of a full turn,
        # include them so the conversation remains coherent.
        if current_turn:
            trimmed = current_turn + trimmed

        return trimmed

    # ------------------------------------------------------------------
    # Skill management
    # ------------------------------------------------------------------

    @property
    def system_prompt(self) -> str:
        if self._skill is not None:
            return self._skill.system_prompt
        return DEFAULT_SYSTEM_PROMPT

    def set_skill(self, skill: "Skill") -> None:
        self._skill = skill
