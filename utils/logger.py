import logging
import os

from rich.logging import RichHandler

_configured = False


def configure_logging(level: str | None = None) -> None:
    """Configure the root logger once. Call this before any get_logger calls."""
    global _configured
    resolved = (level or os.environ.get("LOG_LEVEL", "INFO")).upper()
    os.environ["LOG_LEVEL"] = resolved  # propagate to child loggers
    root = logging.getLogger()
    if not root.handlers:
        handler = RichHandler(rich_tracebacks=True, show_path=False)
        handler.setLevel(resolved)
        root.addHandler(handler)
    root.setLevel(resolved)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    if not _configured:
        configure_logging()
    return logging.getLogger(name)
