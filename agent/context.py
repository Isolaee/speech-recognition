from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TYPE_CHECKING

from llm.base import ToolCall
from utils.logger import get_logger

if TYPE_CHECKING:
    from skills.base import Skill

logger = get_logger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Be concise and clear in your responses."
)


class ConversationContext:
    def __init__(
        self,
        max_turns: int,
        skill: "Skill | None" = None,
        summarizer: "Callable[[list[dict]], str] | None" = None,
    ):
        self.max_turns = max_turns
        self._skill: "Skill | None" = skill
        self._history: list[dict] = []
        self._summarizer = summarizer
        self._summarizing = False

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
    # Summarization / compression
    # ------------------------------------------------------------------

    def _plain_user_turn_indices(self) -> list[int]:
        """Return indices of plain user messages (turn boundaries) in _history."""
        return [
            i for i, msg in enumerate(self._history)
            if msg["role"] == "user" and not isinstance(msg.get("content"), list)
        ]

    def _compress_history(self) -> None:
        """Summarize the oldest messages when turn count exceeds max_turns.

        Old messages are replaced immediately with a placeholder; the actual
        summarization runs in a background thread so it never blocks a response.
        """
        if self._summarizer is None or self._summarizing:
            return

        turn_starts = self._plain_user_turn_indices()
        if len(turn_starts) <= self.max_turns:
            return

        split_idx = turn_starts[-self.max_turns]
        old_messages = self._history[:split_idx]
        recent_messages = self._history[split_idx:]

        if not old_messages:
            return

        # Replace old messages immediately with a placeholder so future calls
        # don't re-trigger compression while the thread is running.
        placeholder = {"role": "user", "content": "[Summarizing previous conversation...]"}
        self._history = [placeholder] + recent_messages
        self._summarizing = True
        logger.debug("Spawning background summarization for %d old messages.", len(old_messages))

        def _run() -> None:
            try:
                summary = self._summarizer(old_messages)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Summarization failed (%s); using fallback.", exc)
                summary = "Previous conversation history omitted."

            summary_msg = {"role": "user", "content": f"[Conversation summary: {summary}]"}
            # Swap placeholder for real summary (identity check is safe here)
            if self._history and self._history[0] is placeholder:
                self._history[0] = summary_msg
            self._summarizing = False
            logger.info("Background summarization complete.")

        threading.Thread(target=_run, daemon=True).start()

    def to_messages(self) -> list[dict]:
        """Return history, compressing old turns into a summary if over max_turns."""
        if self._summarizer:
            self._compress_history()
        return list(self._history)

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
