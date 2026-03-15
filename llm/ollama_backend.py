from collections.abc import Generator

import ollama

from config import OllamaConfig
from llm.base import LLMBackend, LLMResponse, ToolCall
from utils.logger import get_logger

logger = get_logger(__name__)


class OllamaBackend(LLMBackend):
    def __init__(self, config: OllamaConfig):
        self.config = config
        self.client = ollama.Client(host=config.base_url)

    def chat(self, messages: list[dict], tools: list[dict], system: str = "") -> LLMResponse:
        full_messages = self._prepend_system(messages, system)
        response = self.client.chat(
            model=self.config.model,
            messages=full_messages,
            tools=tools or None,
            stream=False,
        )
        return self._parse_response(response)

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
