import threading

from tools.base import BaseTool, ToolResult
from tools.file_ops import _safe_path
from tools.registry import register_tool

MAX_UNDO = 50


@register_tool(backends=["ollama", "claude"])
class TextEditorTool(BaseTool):
    name = "text_editor"
    description = (
        "Edit a text document using voice commands. The document persists across calls "
        "within this session. Use this tool when the user wants to write, edit, draft, "
        "or compose text such as notes, letters, emails, lists, or any document. "
        "Actions: 'new' (start fresh), 'append' (add to end), 'insert' (add at line), "
        "'replace' (find/replace), 'delete' (remove text), 'undo' (revert last change), "
        "'read' (show document), 'save' (write to file). "
        "After any edit action, the tool returns the updated document with line numbers."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "new", "append", "insert", "replace",
                    "delete", "undo", "read", "save",
                ],
                "description": (
                    "'new' creates a blank document (optionally with initial text). "
                    "'append' adds text to the end. "
                    "'insert' adds text at a specific line number. "
                    "'replace' finds and replaces text. "
                    "'delete' removes a line, range of lines, or matching text. "
                    "'undo' reverts the last change. "
                    "'read' shows the current document (optionally a line range). "
                    "'save' writes the document to a file."
                ),
            },
            "text": {
                "type": "string",
                "description": (
                    "The text content. Used by: 'new' (optional initial content), "
                    "'append' (text to add), 'insert' (text to insert)."
                ),
            },
            "line": {
                "type": "integer",
                "description": (
                    "Line number (1-based). Used by: 'insert' (where to insert), "
                    "'delete' (single line to remove), 'read' (start line)."
                ),
            },
            "end_line": {
                "type": "integer",
                "description": (
                    "End line number (1-based, inclusive). Used by: "
                    "'delete' (end of range), 'read' (end of range)."
                ),
            },
            "find": {
                "type": "string",
                "description": "Text to search for. Used by: 'replace' and 'delete' (match-based deletion).",
            },
            "replace_with": {
                "type": "string",
                "description": "Replacement text. Used by: 'replace'.",
            },
            "filename": {
                "type": "string",
                "description": "Filename for 'save' action. Saved within the safe root directory.",
            },
            "name": {
                "type": "string",
                "description": "Optional document name for 'new' action.",
            },
        },
        "required": ["action"],
    }

    def __init__(self):
        self._buffer: str = ""
        self._undo_stack: list[str] = []
        self._document_name: str = "untitled"
        self._lock = threading.Lock()

    # ── public entry point ──────────────────────────────────────────

    def execute(self, action: str, **kwargs) -> ToolResult:
        actions = {
            "new": self._action_new,
            "append": self._action_append,
            "insert": self._action_insert,
            "replace": self._action_replace,
            "delete": self._action_delete,
            "undo": self._action_undo,
            "read": self._action_read,
            "save": self._action_save,
        }
        handler = actions.get(action)
        if not handler:
            return ToolResult(
                success=False, output="",
                error=f"Unknown action '{action}'. Valid: {', '.join(actions)}",
            )
        try:
            return handler(**kwargs)
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    # ── actions ─────────────────────────────────────────────────────

    def _action_new(self, text: str = "", name: str = "untitled", **_) -> ToolResult:
        with self._lock:
            self._undo_stack.clear()
            self._buffer = text
            self._document_name = name
        if text:
            return ToolResult(
                success=True,
                output=f"New document '{name}' created.\n{self._format_buffer()}",
            )
        return ToolResult(
            success=True,
            output=f"New blank document '{name}' created. Start dictating.",
        )

    def _action_append(self, text: str | None = None, **_) -> ToolResult:
        if text is None:
            return ToolResult(success=False, output="", error="'append' requires 'text'.")
        with self._lock:
            self._push_undo()
            if self._buffer and not self._buffer.endswith("\n"):
                self._buffer += "\n"
            self._buffer += text
        return ToolResult(success=True, output=self._format_buffer())

    def _action_insert(self, text: str | None = None, line: int | None = None, **_) -> ToolResult:
        if text is None:
            return ToolResult(success=False, output="", error="'insert' requires 'text'.")
        if line is None:
            return ToolResult(success=False, output="", error="'insert' requires 'line'.")
        with self._lock:
            self._push_undo()
            lines = self._buffer.split("\n") if self._buffer else []
            idx = max(0, min(line - 1, len(lines)))
            for i, new_line in enumerate(text.split("\n")):
                lines.insert(idx + i, new_line)
            self._buffer = "\n".join(lines)
        return ToolResult(success=True, output=self._format_buffer())

    def _action_replace(self, find: str | None = None, replace_with: str = "", **_) -> ToolResult:
        if not find:
            return ToolResult(success=False, output="", error="'replace' requires 'find'.")
        with self._lock:
            count = self._buffer.count(find)
            if count == 0:
                return ToolResult(
                    success=False, output="",
                    error=f"Text not found: '{find}'. No changes made.",
                )
            self._push_undo()
            self._buffer = self._buffer.replace(find, replace_with)
        return ToolResult(
            success=True,
            output=f"Replaced {count} occurrence(s).\n{self._format_buffer()}",
        )

    def _action_delete(
        self, find: str | None = None, line: int | None = None, end_line: int | None = None, **_,
    ) -> ToolResult:
        if find:
            with self._lock:
                if find not in self._buffer:
                    return ToolResult(
                        success=False, output="",
                        error=f"Text not found: '{find}'. No changes made.",
                    )
                self._push_undo()
                self._buffer = self._buffer.replace(find, "")
            return ToolResult(
                success=True,
                output=f"Deleted all occurrences of '{find}'.\n{self._format_buffer()}",
            )
        if line is not None:
            with self._lock:
                lines = self._buffer.split("\n") if self._buffer else []
                start = max(0, line - 1)
                end = min(len(lines), (end_line or line))
                if start >= len(lines):
                    return ToolResult(
                        success=False, output="",
                        error=f"Line {line} is out of range (document has {len(lines)} lines).",
                    )
                self._push_undo()
                deleted = lines[start:end]
                del lines[start:end]
                self._buffer = "\n".join(lines)
            return ToolResult(
                success=True,
                output=f"Deleted: {'; '.join(deleted)}\n{self._format_buffer()}",
            )
        return ToolResult(
            success=False, output="",
            error="'delete' requires either 'find' or 'line'.",
        )

    def _action_undo(self, **_) -> ToolResult:
        with self._lock:
            if not self._undo_stack:
                return ToolResult(success=False, output="", error="Nothing to undo.")
            self._buffer = self._undo_stack.pop()
        return ToolResult(
            success=True,
            output=f"Undone.\n{self._format_buffer()}",
        )

    def _action_read(self, line: int | None = None, end_line: int | None = None, **_) -> ToolResult:
        if not self._buffer:
            return ToolResult(success=True, output="Document is empty.")
        return ToolResult(success=True, output=self._format_buffer(line, end_line))

    def _action_save(self, filename: str | None = None, **_) -> ToolResult:
        fname = filename or f"{self._document_name}.txt"
        try:
            path = _safe_path(fname)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(self._buffer, encoding="utf-8")
        except PermissionError as e:
            return ToolResult(success=False, output="", error=str(e))
        return ToolResult(
            success=True,
            output=f"Document saved to {path} ({len(self._buffer)} bytes).",
        )

    # ── helpers ─────────────────────────────────────────────────────

    def _push_undo(self) -> None:
        self._undo_stack.append(self._buffer)
        if len(self._undo_stack) > MAX_UNDO:
            self._undo_stack.pop(0)

    def _format_buffer(self, start_line: int | None = None, end_line: int | None = None) -> str:
        lines = self._buffer.split("\n") if self._buffer else []
        total = len(lines)
        s = max(0, (start_line or 1) - 1)
        e = min(total, end_line or total)
        numbered = [f"{i + s + 1}: {l}" for i, l in enumerate(lines[s:e])]
        header = f"Document '{self._document_name}' ({total} lines):"
        return header + "\n" + "\n".join(numbered)
