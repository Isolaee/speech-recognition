from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from llm.base import ToolCall
from tools.base import BaseTool, ToolResult
from tools.registry import ToolRegistry
from utils.logger import get_logger

logger = get_logger(__name__)


class ToolExecutor:
    def __init__(self, registry: ToolRegistry, timeout: int = 15):
        self.registry = registry
        self.timeout = timeout

    def execute_batch(
        self, tool_calls: list[ToolCall], backend: str = "ollama"
    ) -> list[tuple[ToolCall, ToolResult]]:
        results = []
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self._run, tc, backend): tc
                for tc in tool_calls
            }
            for future, tc in futures.items():
                try:
                    result = future.result(timeout=self.timeout)
                except FuturesTimeoutError:
                    logger.warning("Tool '%s' timed out after %ss", tc.name, self.timeout)
                    result = ToolResult(
                        success=False,
                        output="",
                        error=f"Tool timed out after {self.timeout} seconds",
                    )
                except Exception as e:
                    logger.error("Tool '%s' raised unexpected error: %s", tc.name, e)
                    result = ToolResult(success=False, output="", error=str(e))
                results.append((tc, result))
        return results

    def _run(self, tc: ToolCall, backend: str) -> ToolResult:
        try:
            tool: BaseTool = self.registry.lookup(tc.name)
        except KeyError as e:
            return ToolResult(success=False, output="", error=str(e))
        try:
            logger.debug("Executing tool '%s' with args %s", tc.name, tc.arguments)
            result = tool.execute(**tc.arguments)
            logger.debug("Tool '%s' result: %s", tc.name, result.output[:120])
            return result
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
