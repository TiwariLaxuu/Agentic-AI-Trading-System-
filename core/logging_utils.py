"""In-memory ring-buffer logging so the Logs / System Status page can display
recent activity without needing a log file or external aggregator."""
from __future__ import annotations

import logging
from collections import deque
from datetime import datetime

_BUFFER: deque[dict] = deque(maxlen=500)

LEVEL_ORDER = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}


class BufferHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        _BUFFER.append({
            "time": datetime.fromtimestamp(record.created).strftime("%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        })


_root = logging.getLogger("ai_trading_engine")
if not _root.handlers:
    _root.setLevel(logging.INFO)
    _root.addHandler(BufferHandler())
    _root.propagate = False


def get_logger(name: str) -> logging.Logger:
    return _root.getChild(name)


def get_logs() -> list[dict]:
    return list(_BUFFER)


def clear_logs() -> None:
    _BUFFER.clear()
