from __future__ import annotations

from collections.abc import Generator
from typing import TYPE_CHECKING

import ollama

from config import OllamaConfig
from llm.base import LLMBackend, LLMResponse, ToolCall
from utils.logger import get_logger

if TYPE_CHECKING:
    from agent.tool_executor import ToolExecutor

logger = get_logger(__name__)


class OllamaBackend(LLMBackend):
    def __init__(self, config: OllamaConfig, tool_executor: ToolExecutor | None = None):
        self.config = config
        self.client = ollama.Client(host=config.base_url)
        self.tool_executor = tool_executor

    def chat(self, messages: list[dict], tools: list[dict], system: str = "", model_override: str | None = None) -> LLMResponse:
        full_messages = self._prepend_system(messages, system)
        model = model_override or self.config.model

        while True:
            response = self.client.chat(
                model=model,
                messages=full_messages,
                tools=tools or None,
                stream=False,
            )
            parsed = self._parse_response(response)

            if not parsed.tool_calls or self.tool_executor is None:
                return parsed

            # Append assistant tool-call message
            full_messages.append({
                "role": "assistant",
                "content": response.message.content or "",
                "tool_calls": [
                    {
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments,
                        }
                    }
                    for tc in parsed.tool_calls
                ],
            })

            # Execute tools and append results
            pairs = self.tool_executor.execute_batch(parsed.tool_calls, backend="ollama")
            for tc, result in pairs:
                full_messages.append({
                    "role": "tool",
                    "content": result.output if result.success else f"Error: {result.error}",
                })

            logger.debug("Tool loop: appended %d results, continuing", len(pairs))

    def stream_chat(self, messages: list[dict], tools: list[dict], system: str = "") -> Generator[str, None, None]:
        full_messages = self._prepend_system(messages, system)
        stream = self.client.chat(
            model=self.config.model,
            messages=full_messages,
            tools=tools or None,
            stream=True,
        )
        for chunk in stream:
            content = chunk.message.content
            if content:
                yield content

    def _prepend_system(self, messages: list[dict], system: str) -> list[dict]:
        if not system:
            return messages
        return [{"role": "system", "content": system}] + messages

    def _parse_response(self, response) -> LLMResponse:
        message = response.message
        text = message.content or None

        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(ToolCall(
                    id=str(id(tc)),
                    name=tc.function.name,
                    arguments=dict(tc.function.arguments),
                ))

        usage = {}
        if hasattr(response, "prompt_eval_count"):
            usage["prompt_tokens"] = response.prompt_eval_count
        if hasattr(response, "eval_count"):
            usage["completion_tokens"] = response.eval_count

        return LLMResponse(text=text, tool_calls=tool_calls, usage=usage)
