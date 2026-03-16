from __future__ import annotations

import queue
import re
import string
import threading
import time
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
        self._utterance_queue: queue.Queue[str] = queue.Queue()

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
        if not self.no_voice:
            t = threading.Thread(target=self._audio_capture_loop, daemon=True)
            t.start()
        try:
            while True:
                if self.no_voice:
                    try:
                        text = input("You: ").strip()
                    except EOFError:
                        break
                    if text:
                        self._handle_utterance(text)
                else:
                    try:
                        text = self._utterance_queue.get(timeout=0.1)
                        self._handle_utterance(text)
                    except queue.Empty:
                        continue
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

    def _is_skip_phrase(self, text: str) -> bool:
        phrase = self._normalize(self.config.agent.skip_phrase)
        return phrase in self._normalize(text)

    def _is_stop_phrase(self, text: str) -> bool:
        phrase = self._normalize(self.config.agent.stop_phrase)
        return phrase in self._normalize(text)

    def _speak_sentences_from_stream(self, gen: Generator[str, None, None], voice_override: str | None = None) -> str:
        """Consume a streaming text generator, speaking each sentence as it completes.

        Synthesis of sentence N+1 runs in a background thread while sentence N plays,
        so the gap between sentences is near-zero.
        """
        import numpy as np
        import sounddevice as sd
        from concurrent.futures import Future, ThreadPoolExecutor

        full_text = ""
        buffer = ""
        _sentence_end = re.compile(r"[.!?]\s")

        executor = ThreadPoolExecutor(max_workers=1)

        def _synthesize(text: str) -> tuple:
            chunks = list(self.tts.voice.synthesize(text))
            if not chunks:
                return None, None
            return np.concatenate([c.audio_float_array for c in chunks]), chunks[0].sample_rate

        def _play(audio, rate: int) -> bool:
            """Play audio. Returns False if user interrupted."""
            sd.play(audio, samplerate=rate)
            while sd.get_stream().active:
                try:
                    heard = self._utterance_queue.get_nowait()
                    sd.stop()
                    if not self._is_skip_phrase(heard) and not self._is_stop_phrase(heard):
                        self._utterance_queue.put(heard)
                    return False
                except queue.Empty:
                    pass
                time.sleep(0.05)
            sd.wait()
            return True

        pending: Future | None = None
        interrupted = False

        def queue_sentence(sentence: str) -> bool:
            """Submit synthesis for sentence; play the previously queued one. Returns False if interrupted."""
            nonlocal pending
            future = executor.submit(_synthesize, sentence)
            if pending is not None:
                audio, rate = pending.result()
                if audio is not None and not _play(audio, rate):
                    pending = future
                    return False
            pending = future
            return True

        for chunk in gen:
            full_text += chunk
            buffer += chunk
            while True:
                m = _sentence_end.search(buffer)
                if not m:
                    break
                sentence = buffer[:m.end() - 1].strip()
                buffer = buffer[m.end():]
                if sentence and not interrupted:
                    if not queue_sentence(sentence):
                        interrupted = True

        if buffer.strip() and not interrupted:
            queue_sentence(buffer.strip())

        if pending is not None and not interrupted:
            audio, rate = pending.result()
            if audio is not None:
                _play(audio, rate)

        executor.shutdown(wait=False)
        return full_text

    def _speak_with_skip(self, text: str, voice_override: str | None = None) -> None:
        """Speak text, interruptible mid-playback.

        - skip / stop phrase  → stop TTS, discard utterance, stay in chat mode
        - any other utterance → stop TTS, re-queue as refinement input
        """
        if not self.tts:
            return

        import sounddevice as sd
        import numpy as np

        chunks = list(self.tts.voice.synthesize(text))
        if not chunks:
            return
        sample_rate = chunks[0].sample_rate
        audio = np.concatenate([c.audio_float_array for c in chunks])

        sd.play(audio, samplerate=sample_rate)
        interrupted = False
        while sd.get_stream().active:
            try:
                heard = self._utterance_queue.get_nowait()
                sd.stop()
                interrupted = True
                if self._is_skip_phrase(heard):
                    logger.info("Skip — TTS stopped, discarding.")
                elif self._is_stop_phrase(heard):
                    logger.info("Stop — TTS stopped, returning to chat mode.")
                else:
                    logger.info("Refinement mid-TTS: '%s' — re-queuing.", heard)
                    self._utterance_queue.put(heard)
                break
            except queue.Empty:
                pass
            time.sleep(0.05)
        if not interrupted:
            sd.wait()

    def _audio_capture_loop(self) -> None:
        """Background thread: continuously transcribe mic audio into _utterance_queue."""
        for audio in self.audio_input.stream_utterances():
            text = self.stt.transcribe(audio)
            if not text:
                continue
            logger.info("Transcribed: %s", text)
            if self._chat_mode:
                self._utterance_queue.put(text)
            elif self._wake_word_detected(text):
                self._chat_mode = True
                logger.info("Wake word detected — entering chat mode.")
                stripped = self._strip_wake_word(text)
                if stripped:
                    self._utterance_queue.put(stripped)
            else:
                logger.debug("Wake word not detected; ignoring utterance.")

    def _handle_utterance(self, text: str) -> None:
        if self._is_sleep_phrase(text):
            self._chat_mode = False
            msg = "Going to sleep. Say the wake word when you need me."
            logger.info("Sleep command received — exiting chat mode.")
            if self.no_voice:
                print(f"Agent: {msg}")
            else:
                self._speak_with_skip(msg)
            return

        if self._is_skill_switch(text):
            self._handle_skill_switch(text)
            return

        self.context.add_user(text)

        if self.no_voice:
            response = self.router.chat(
                messages=self.context.to_messages(),
                skill=self._current_skill,
                tts=self.tts,
                system=self.context.system_prompt,
            )
            response_text = response.text or ""
            self.context.add_assistant(response_text)
            print(f"Agent: {response_text}")
        else:
            gen = self.router.stream_chat(
                messages=self.context.to_messages(),
                skill=self._current_skill,
                system=self.context.system_prompt,
            )
            response_text = self._speak_sentences_from_stream(gen, voice_override=self._current_skill.tts_voice)
            self.context.add_assistant(response_text)

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
                else:
                    self._speak_with_skip(msg)
                return
