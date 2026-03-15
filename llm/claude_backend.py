import os

import anthropic

from config import ClaudeConfig
from llm.base import LLMBackend, LLMResponse, ToolCall
from utils.logger import get_logger

logger = get_logger(__name__)


class ClaudeBackend(LLMBackend):
    def __init__(self, config: ClaudeConfig):
        self.config = config
        api_key = os.environ.get(config.api_key_env)
        self.client = anthropic.Anthropic(api_key=api_key)

    def chat(self, messages: list[dict], tools: list[dict], system: str = "") -> LLMResponse:
        anthropic_tools = [self._convert_tool_schema(t) for t in tools] if tools else []

        response = self.client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            system=system or anthropic.NOT_GIVEN,
            messages=messages,
            tools=anthropic_tools or anthropic.NOT_GIVEN,
        )

        # Handle multi-turn tool loop
        current_messages = list(messages)
        while response.stop_reason == "tool_use":
            tool_calls = []
            tool_use_blocks = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_use_blocks.append(block)
                    tool_calls.append(ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=dict(block.input),
                    ))

            # Append assistant message with tool use blocks
            current_messages.append({"role": "assistant", "content": response.content})

            # Caller must provide tool results; since we don't have an executor here,
            # we break out and return the tool calls for the caller to handle.
            # But per the plan, ClaudeBackend handles the loop internally once
            # a ToolExecutor is wired in (Step 19). For now, return partial response.
            logger.debug("Claude requested tool calls: %s", [tc.name for tc in tool_calls])
            return LLMResponse(
                text=None,
                tool_calls=tool_calls,
                usage=self._extract_usage(response),
            )

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
