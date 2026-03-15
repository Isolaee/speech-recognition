"""
Integration tests for Phase 7.

Covers:
  1. OllamaBackend tool loop: first call returns a tool call, second returns final text.
  2. Escalation routing: Ollama raises EscalationRequested → Claude backend is called.
  3. Orchestrator (no-voice): _handle_utterance processes text and prints the response.

Run with:  python test_integration.py
"""
import io
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import MagicMock

# Ensure project root is on sys.path when run directly
import os
sys.path.insert(0, os.path.dirname(__file__))

from llm.base import LLMResponse, ToolCall
from tools.base import EscalationRequested


# ---------------------------------------------------------------------------
# Helper: build a minimal fake ollama response
# ---------------------------------------------------------------------------

def _fake_ollama_response(content: str, tool_calls=None):
    resp = MagicMock()
    resp.message.content = content
    resp.message.tool_calls = tool_calls
    return resp


# ---------------------------------------------------------------------------
# Test 1 – OllamaBackend tool loop
# ---------------------------------------------------------------------------

class TestOllamaToolLoop(unittest.TestCase):
    """OllamaBackend executes a tool and loops back to the LLM for the final answer."""

    def test_tool_loop_calls_llm_twice(self):
        import tools.calculator  # register the calculator tool
        from agent.tool_executor import ToolExecutor
        from config import OllamaConfig
        from llm.ollama_backend import OllamaBackend
        from tools.registry import registry

        executor = ToolExecutor(registry, timeout=5)
        backend = OllamaBackend(OllamaConfig(), tool_executor=executor)
        backend.client = MagicMock()

        # First response: tool call for calculator
        tc_mock = MagicMock()
        tc_mock.function.name = "calculator"
        tc_mock.function.arguments = {"expression": "3 * 7"}
        first = _fake_ollama_response("", tool_calls=[tc_mock])

        # Second response: final text
        second = _fake_ollama_response("3 times 7 is 21.", tool_calls=None)

        backend.client.chat.side_effect = [first, second]

        result = backend.chat(
            messages=[{"role": "user", "content": "What is 3 * 7?"}],
            tools=[],
            system="",
        )

        self.assertEqual(result.text, "3 times 7 is 21.")
        self.assertEqual(backend.client.chat.call_count, 2, "LLM should be called twice (tool loop)")

    def test_no_tool_call_returns_directly(self):
        from config import OllamaConfig
        from llm.ollama_backend import OllamaBackend

        backend = OllamaBackend(OllamaConfig())
        backend.client = MagicMock()
        backend.client.chat.return_value = _fake_ollama_response("Hello!", tool_calls=None)

        result = backend.chat(
            messages=[{"role": "user", "content": "Hi"}],
            tools=[],
            system="",
        )

        self.assertEqual(result.text, "Hello!")
        self.assertEqual(backend.client.chat.call_count, 1)


# ---------------------------------------------------------------------------
# Test 2 – Escalation routing
# ---------------------------------------------------------------------------

class TestEscalationRouting(unittest.TestCase):
    """LLMRouter escalates to Claude when OllamaBackend raises EscalationRequested."""

    def _make_router(self):
        from config import ClaudeConfig, EscalationConfig, OllamaConfig
        from llm.claude_backend import ClaudeBackend
        from llm.ollama_backend import OllamaBackend
        from llm.router import LLMRouter
        from tools.registry import registry

        ollama = OllamaBackend(OllamaConfig())
        claude = ClaudeBackend(ClaudeConfig())
        router = LLMRouter(ollama, claude, registry, EscalationConfig(enabled=True))
        return router, ollama, claude

    def _default_skill(self):
        from skills.base import Skill
        return Skill(
            name="default",
            description="",
            system_prompt="Be helpful.",
            enabled_tools=[],
            always_escalate=False,
        )

    def test_escalation_calls_claude(self):
        router, ollama, claude = self._make_router()

        ollama.chat = MagicMock(
            side_effect=EscalationRequested(
                reason="too complex",
                refined_prompt="Explain quantum entanglement simply.",
                context_summary="User asked about quantum physics.",
            )
        )
        expected = LLMResponse(text="Quantum entanglement is...", tool_calls=[], usage={})
        claude.chat = MagicMock(return_value=expected)

        result = router.chat(
            messages=[{"role": "user", "content": "Explain quantum entanglement"}],
            skill=self._default_skill(),
            tts=None,
            system="Be helpful.",
        )

        self.assertTrue(claude.chat.called, "Claude should be called on escalation")
        self.assertEqual(result.text, "Quantum entanglement is...")

    def test_always_escalate_skips_ollama(self):
        router, ollama, claude = self._make_router()

        from skills.base import Skill
        skill = Skill(
            name="coding_assistant",
            description="",
            system_prompt="You are a coding assistant.",
            enabled_tools=[],
            always_escalate=True,
        )
        ollama.chat = MagicMock()
        expected = LLMResponse(text="Here is the code.", tool_calls=[], usage={})
        claude.chat = MagicMock(return_value=expected)

        result = router.chat(
            messages=[{"role": "user", "content": "Write a sort function"}],
            skill=skill,
            tts=None,
            system="You are a coding assistant.",
        )

        self.assertFalse(ollama.chat.called, "Ollama should not be called when always_escalate=True")
        self.assertTrue(claude.chat.called)
        self.assertEqual(result.text, "Here is the code.")


# ---------------------------------------------------------------------------
# Test 3 – Orchestrator text mode
# ---------------------------------------------------------------------------

class TestOrchestratorTextMode(unittest.TestCase):
    """Orchestrator in no-voice mode processes input and prints the response."""

    def _make_orchestrator(self):
        from agent.orchestrator import Orchestrator
        from config import Config
        return Orchestrator(Config(), no_voice=True)

    def test_handle_utterance_prints_response(self):
        orchestrator = self._make_orchestrator()
        orchestrator.router.chat = MagicMock(
            return_value=LLMResponse(text="Hello! How can I help?", tool_calls=[], usage={})
        )

        buf = io.StringIO()
        with redirect_stdout(buf):
            orchestrator._handle_utterance("Hello")

        output = buf.getvalue()
        self.assertIn("Hello! How can I help?", output)

    def test_response_added_to_context(self):
        orchestrator = self._make_orchestrator()
        orchestrator.router.chat = MagicMock(
            return_value=LLMResponse(text="Sure, I can help.", tool_calls=[], usage={})
        )

        with redirect_stdout(io.StringIO()):
            orchestrator._handle_utterance("Can you help me?")

        messages = orchestrator.context.to_messages()
        self.assertEqual(messages[-1]["content"], "Sure, I can help.")
        self.assertEqual(messages[-1]["role"], "assistant")

    def test_skill_switch_detected(self):
        orchestrator = self._make_orchestrator()
        orchestrator.router.chat = MagicMock()

        buf = io.StringIO()
        with redirect_stdout(buf):
            orchestrator._handle_utterance("switch to home_assistant skill")

        self.assertFalse(orchestrator.router.chat.called, "Router should not be called for skill switches")
        self.assertEqual(orchestrator._current_skill.name, "home_assistant")

    def test_unknown_skill_switch_does_not_crash(self):
        orchestrator = self._make_orchestrator()

        buf = io.StringIO()
        with redirect_stdout(buf):
            orchestrator._handle_utterance("switch to nonexistent skill")

        output = buf.getvalue()
        self.assertIn("nonexistent", output)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
