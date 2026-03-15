import logging
import os

from rich.logging import RichHandler


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        level = os.environ.get("LOG_LEVEL", "INFO").upper()
        logger.setLevel(level)
        handler = RichHandler(rich_tracebacks=True, show_path=False)
        handler.setLevel(level)
        logger.addHandler(handler)
    return logger
