from __future__ import annotations

import re
import string
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
import tools.whatsup

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
        self._chat_mode: bool = False  # True = no wake word required

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

    @staticmethod
    def _normalize(text: str) -> str:
        """Lowercase and strip punctuation for fuzzy phrase matching."""
        return text.lower().translate(str.maketrans("", "", string.punctuation))

    def _wake_word_detected(self, text: str) -> bool:
        """Return True if wake word is disabled or found at the start of text."""
        if not self.config.agent.wake_word_enabled:
            return True
        wake = self._normalize(self.config.agent.wake_word)
        return self._normalize(text).startswith(wake)

    def _strip_wake_word(self, text: str) -> str:
        """Remove the wake word prefix from text (case-insensitive)."""
        wake = self.config.agent.wake_word
        # Find end of wake word in original text by comparing normalized prefix
        norm_text = self._normalize(text)
        norm_wake = self._normalize(wake)
        if norm_text.startswith(norm_wake):
            # Walk forward in original text consuming the same number of non-punct chars
            consumed = 0
            i = 0
            while i < len(text) and consumed < len(norm_wake):
                if text[i] not in string.punctuation and text[i] != " ":
                    consumed += 1
                i += 1
            return text[i:].lstrip(" ,.")
        return text

    def _is_sleep_phrase(self, text: str) -> bool:
        phrase = self._normalize(self.config.agent.sleep_phrase)
        return phrase in self._normalize(text)

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
                if self._chat_mode:
                    yield text
                elif self._wake_word_detected(text):
                    self._chat_mode = True
                    logger.info("Wake word detected — entering chat mode.")
                    stripped = self._strip_wake_word(text)
                    if stripped:
                        yield stripped
                else:
                    logger.debug("Wake word not detected; ignoring utterance.")

    def _handle_utterance(self, text: str) -> None:
        if self._is_sleep_phrase(text):
            self._chat_mode = False
            msg = "Going to sleep. Say the wake word when you need me."
            logger.info("Sleep command received — exiting chat mode.")
            if self.no_voice:
                print(f"Agent: {msg}")
            elif self.tts:
                self.tts.speak(msg)
            return

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
