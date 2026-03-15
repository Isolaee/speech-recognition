from __future__ import annotations

import re
from collections.abc import Generator

from config import Config
from agent.context import ConversationContext
from agent.tool_executor import ToolExecutor
from llm.claude_backend import ClaudeBackend
from llm.ollama_backend import OllamaBackend
from llm.router import LLMRouter
from skills.loader import SkillLoader
from tools.registry import registry
from utils.logger import get_logger

# Import tool modules to trigger @register_tool decorator registration
import tools.calculator
import tools.time_weather
import tools.system_info
import tools.file_ops
import tools.web_search

logger = get_logger(__name__)

_SKILL_SWITCH_RE = [
    re.compile(r"switch to (\w[\w\s]*?) (?:assistant|mode|skill)", re.IGNORECASE),
    re.compile(r"use (\w[\w\s]*?) (?:assistant|mode|skill)", re.IGNORECASE),
    re.compile(r"activate (\w[\w\s]*?) (?:assistant|mode|skill)", re.IGNORECASE),
]


class Orchestrator:
    def __init__(self, config: Config, no_voice: bool = False):
        self.config = config
        self.no_voice = no_voice

        self.skill_loader = SkillLoader("skills/definitions")

        self.executor = ToolExecutor(registry, timeout=config.agent.tool_timeout_seconds)
        self.ollama = OllamaBackend(config.ollama, tool_executor=self.executor)
        self.claude = ClaudeBackend(config.claude, tool_executor=self.executor)
        self.router = LLMRouter(self.ollama, self.claude, registry, config.escalation)

        initial_skill = self.skill_loader.load(config.agent.active_skill)
        self.context = ConversationContext(
            max_turns=config.agent.history_max_turns,
            skill=initial_skill,
            summarizer=self._make_summarizer(),
        )
        self._current_skill = initial_skill

        self.audio_input = None
        self.stt = None
        self.tts = None

        if not no_voice:
            from voice.input import AudioInput
            from voice.stt import FasterWhisperBackend
            from voice.tts import PiperBackend

            self.audio_input = AudioInput(config.vad)
            self.stt = FasterWhisperBackend(config.stt)
            self.tts = PiperBackend(config.tts)

    def run(self) -> None:
        logger.info(
            "Orchestrator starting (no_voice=%s, skill=%s)",
            self.no_voice,
            self._current_skill.name,
        )
        try:
            for text in self._utterance_source():
                self._handle_utterance(text)
        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
            print("\nGoodbye.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _wake_word_detected(self, text: str) -> bool:
        """Return True if wake word is disabled or found at the start of text."""
        if not self.config.agent.wake_word_enabled:
            return True
        wake = self.config.agent.wake_word.lower()
        return text.lower().startswith(wake)

    def _strip_wake_word(self, text: str) -> str:
        """Remove the wake word prefix from text (case-insensitive)."""
        wake = self.config.agent.wake_word.lower()
        lower = text.lower()
        if lower.startswith(wake):
            return text[len(wake):].lstrip(" ,")
        return text

    def _utterance_source(self) -> Generator[str, None, None]:
        if self.no_voice:
            while True:
                try:
                    text = input("You: ").strip()
                except EOFError:
                    break
                if text:
                    yield text
        else:
            for audio in self.audio_input.stream_utterances():
                text = self.stt.transcribe(audio)
                if not text:
                    continue
                logger.info("Transcribed: %s", text)
                if not self._wake_word_detected(text):
                    logger.debug("Wake word not detected; ignoring utterance.")
                    continue
                yield self._strip_wake_word(text)

    def _handle_utterance(self, text: str) -> None:
        if self._is_skill_switch(text):
            self._handle_skill_switch(text)
            return

        self.context.add_user(text)
        response = self.router.chat(
            messages=self.context.to_messages(),
            skill=self._current_skill,
            tts=self.tts,
            system=self.context.system_prompt,
        )
        response_text = response.text or ""
        self.context.add_assistant(response_text)

        if self.no_voice:
            print(f"Agent: {response_text}")
        elif self.tts and response_text:
            self.tts.speak(response_text, voice_override=self._current_skill.tts_voice)

    # ------------------------------------------------------------------
    # Skill management
    # ------------------------------------------------------------------

    def _make_summarizer(self):
        """Return a callable that summarizes a message list using Ollama."""
        ollama = self.ollama

        def summarize(messages: list[dict]) -> str:
            summary_messages = list(messages) + [{
                "role": "user",
                "content": (
                    "Please provide a brief 2-3 sentence summary of the above "
                    "conversation, capturing the main topics and any important details."
                ),
            }]
            response = ollama.chat(
                summary_messages,
                tools=[],
                system="You are a helpful assistant that summarizes conversations concisely.",
            )
            return response.text or "Previous conversation history."

        return summarize

    def set_skill(self, name: str) -> None:
        skill = self.skill_loader.load(name)
        self._current_skill = skill
        self.context.set_skill(skill)
        logger.info("Switched to skill '%s'", name)

    def _is_skill_switch(self, text: str) -> bool:
        return any(pattern.search(text) for pattern in _SKILL_SWITCH_RE)

    def _handle_skill_switch(self, text: str) -> None:
        for pattern in _SKILL_SWITCH_RE:
            m = pattern.search(text)
            if m:
                raw_name = m.group(1).strip().lower().replace(" ", "_")
                try:
                    self.set_skill(raw_name)
                    msg = f"Switched to {self._current_skill.name} skill."
                except ValueError:
                    msg = f"Sorry, I don't have a skill named '{raw_name}'."
                    logger.warning("Skill switch failed: unknown skill '%s'", raw_name)

                if self.no_voice:
                    print(f"Agent: {msg}")
                elif self.tts:
                    self.tts.speak(msg)
                return
