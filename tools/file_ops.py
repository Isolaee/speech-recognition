import os
from pathlib import Path

from tools.base import BaseTool, ToolResult
from tools.registry import register_tool

# Safe root: only allow access within the user's home directory by default.
# Can be overridden by setting FILE_OPS_ROOT env var.
_SAFE_ROOT = Path(os.environ.get("FILE_OPS_ROOT", Path.home())).resolve()


def _safe_path(path: str) -> Path:
    resolved = (_SAFE_ROOT / path).resolve()
    if not str(resolved).startswith(str(_SAFE_ROOT)):
        raise PermissionError(f"Path '{path}' is outside the allowed root '{_SAFE_ROOT}'")
    return resolved


@register_tool(backends=["ollama", "claude"])
class ReadFileTool(BaseTool):
    name = "read_file"
    description = "Read the contents of a file. Path is relative to the safe root directory."
    parameters_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative file path to read"},
        },
        "required": ["path"],
    }

    def execute(self, path: str, **kwargs) -> ToolResult:
        try:
            content = _safe_path(path).read_text()
            return ToolResult(success=True, output=content)
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


@register_tool(backends=["ollama", "claude"])
class WriteFileTool(BaseTool):
    name = "write_file"
    description = "Write content to a file. Path is relative to the safe root directory."
    parameters_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative file path to write"},
            "content": {"type": "string", "description": "Content to write"},
        },
        "required": ["path", "content"],
    }

    def execute(self, path: str, content: str, **kwargs) -> ToolResult:
        try:
            target = _safe_path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
            return ToolResult(success=True, output=f"Written {len(content)} bytes to {path}")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


@register_tool(backends=["ollama", "claude"])
class ListDirectoryTool(BaseTool):
    name = "list_directory"
    description = "List files and directories at a path. Path is relative to the safe root."
    parameters_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative directory path to list"},
        },
        "required": ["path"],
    }

    def execute(self, path: str = ".", **kwargs) -> ToolResult:
        try:
            target = _safe_path(path)
            entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name))
            lines = [
                f"{'[dir] ' if e.is_dir() else '[file]'} {e.name}"
                for e in entries
            ]
            return ToolResult(success=True, output="\n".join(lines) or "(empty)")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
