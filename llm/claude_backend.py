from __future__ import annotations

import os
from typing import TYPE_CHECKING

import anthropic

from config import ClaudeConfig
from llm.base import LLMBackend, LLMResponse, ToolCall
from utils.logger import get_logger

if TYPE_CHECKING:
    from agent.tool_executor import ToolExecutor

logger = get_logger(__name__)


class ClaudeBackend(LLMBackend):
    def __init__(self, config: ClaudeConfig, tool_executor: ToolExecutor | None = None):
        self.config = config
        self.tool_executor = tool_executor
        api_key = os.environ.get(config.api_key_env)
        self.client = anthropic.Anthropic(api_key=api_key)

    def chat(self, messages: list[dict], tools: list[dict], system: str = "") -> LLMResponse:
        anthropic_tools = [self._convert_tool_schema(t) for t in tools] if tools else []
        current_messages = list(messages)

        while True:
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                system=system or anthropic.NOT_GIVEN,
                messages=current_messages,
                tools=anthropic_tools or anthropic.NOT_GIVEN,
            )

            if response.stop_reason != "tool_use" or self.tool_executor is None:
                break

            # Extract tool calls from the response
            tool_calls = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_calls.append(ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=dict(block.input),
                    ))

            logger.debug("Claude tool loop: executing %s", [tc.name for tc in tool_calls])

            # Append assistant tool-use message
            current_messages.append({"role": "assistant", "content": response.content})

            # Execute tools and append results
            pairs = self.tool_executor.execute_batch(tool_calls, backend="claude")
            tool_result_content = [
                {
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": result.output if result.success else f"Error: {result.error}",
                }
                for tc, result in pairs
            ]
            current_messages.append({"role": "user", "content": tool_result_content})
            logger.debug("Claude tool loop: %d results appended, continuing", len(pairs))

        text = None
        for block in response.content:
            if hasattr(block, "text"):
                text = block.text
                break

        return LLMResponse(
            text=text,
            tool_calls=[],
            usage=self._extract_usage(response),
        )

    def _convert_tool_schema(self, tool: dict) -> dict:
        """Convert OpenAI-style tool schema to Anthropic format."""
        converted = {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "input_schema": tool.get("parameters", tool.get("input_schema", {"type": "object", "properties": {}})),
        }
        return converted

    def _extract_usage(self, response) -> dict:
        if not hasattr(response, "usage"):
            return {}
        return {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
        }
