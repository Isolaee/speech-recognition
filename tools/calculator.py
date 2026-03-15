from simpleeval import simple_eval, EvalWithCompoundTypes, InvalidExpression

from tools.base import BaseTool, ToolResult
from tools.registry import register_tool


@register_tool(backends=["ollama", "claude"])
class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Safely evaluate a mathematical expression and return the result."
    parameters_schema = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "A mathematical expression to evaluate, e.g. '2 + 2' or 'sqrt(16)'",
            }
        },
        "required": ["expression"],
    }

    def execute(self, expression: str, **kwargs) -> ToolResult:
        try:
            result = simple_eval(expression)
            return ToolResult(success=True, output=str(result))
        except (InvalidExpression, Exception) as e:
            return ToolResult(success=False, output="", error=str(e))
