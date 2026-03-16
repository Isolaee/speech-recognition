import json
import webbrowser
from pathlib import Path
from urllib.parse import quote

from tools.base import BaseTool, ToolResult
from tools.registry import register_tool

CONTACTS_FILE = Path.home() / ".config" / "whatsup" / "contacts.json"


@register_tool(backends=["ollama", "claude"])
class WhatsUpTool(BaseTool):
    name = "whatsup"
    description = (
        "Interact with WhatsApp via WhatsApp Web: open the app, list saved contacts, "
        "send a message, or initiate a call. Actions: 'open', 'contacts', 'send_message', 'call'."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["open", "contacts", "send_message", "call"],
                "description": (
                    "'open' launches WhatsApp Web, 'contacts' lists saved contacts, "
                    "'send_message' opens a chat with a pre-filled message, "
                    "'call' opens the chat to start a voice/video call."
                ),
            },
            "contact": {
                "type": "string",
                "description": (
                    "Contact name or alias from saved contacts. "
                    "Required for 'send_message' and 'call'."
                ),
            },
            "message": {
                "type": "string",
                "description": "Message text to send. Required for 'send_message'.",
            },
        },
        "required": ["action"],
    }

    def _load_contacts(self) -> dict[str, dict]:
        """Load contacts and normalize to {name: {phone, aliases}} format."""
        if not CONTACTS_FILE.exists():
            return {}
        try:
            raw = json.loads(CONTACTS_FILE.read_text())
        except Exception:
            return {}
        normalized = {}
        for name, value in raw.items():
            if isinstance(value, str):
                normalized[name] = {"phone": value, "aliases": []}
            elif isinstance(value, dict):
                normalized[name] = {
                    "phone": value.get("phone", ""),
                    "aliases": [a.lower() for a in value.get("aliases", [])],
                }
        return normalized

    def _find_contact(self, query: str) -> tuple[str, str] | None:
        """Return (canonical_name, phone) for a query matching name or alias, or None."""
        q = query.lower()
        for name, info in self._load_contacts().items():
            if name.lower() == q or q in info["aliases"]:
                return name, info["phone"]
        return None

    def execute(self, action: str, contact: str = "", message: str = "", **kwargs) -> ToolResult:
        if action == "open":
            webbrowser.open("https://web.whatsapp.com")
            return ToolResult(success=True, output="WhatsApp Web opened in your browser.")

        if action == "contacts":
            contacts = self._load_contacts()
            if not contacts:
                return ToolResult(
                    success=True,
                    output=f"No contacts saved yet. Create {CONTACTS_FILE} with entries.",
                )
            lines = []
            for name, info in sorted(contacts.items()):
                alias_str = f"  (aliases: {', '.join(info['aliases'])})" if info["aliases"] else ""
                lines.append(f"{name}: {info['phone']}{alias_str}")
            return ToolResult(success=True, output="\n".join(lines), metadata={"contacts": contacts})

        if action == "send_message":
            if not contact:
                return ToolResult(success=False, output="", error="'contact' is required for send_message.")
            if not message:
                return ToolResult(success=False, output="", error="'message' is required for send_message.")
            result = self._find_contact(contact)
            if not result:
                return ToolResult(success=False, output="", error=f"'{contact}' is not in your contacts. Add them to {CONTACTS_FILE} first.")
            canonical, phone = result
            url = f"https://wa.me/{phone.lstrip('+').replace(' ', '')}?text={quote(message)}"
            webbrowser.open(url)
            return ToolResult(
                success=True,
                output=f"WhatsApp Web opened with pre-filled message to {canonical}. Press Send in the browser.",
            )

        if action == "call":
            if not contact:
                return ToolResult(success=False, output="", error="'contact' is required for call.")
            result = self._find_contact(contact)
            if not result:
                return ToolResult(success=False, output="", error=f"'{contact}' is not in your contacts. Add them to {CONTACTS_FILE} first.")
            canonical, phone = result
            url = f"https://wa.me/{phone.lstrip('+').replace(' ', '')}"
            webbrowser.open(url)
            return ToolResult(
                success=True,
                output=f"WhatsApp Web opened for {canonical}. Use the call button in the chat to start a voice or video call.",
            )

        return ToolResult(
            success=False,
            output="",
            error=f"Unknown action '{action}'. Valid actions: open, contacts, send_message, call.",
        )
