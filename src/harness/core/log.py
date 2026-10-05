"""Structured (JSON-lines) logging."""
from __future__ import annotations

import json
import logging
import sys
import time


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        d = {"t": time.strftime("%H:%M:%S", time.localtime(record.created)), "lvl": record.levelname,
             "name": record.name, "msg": record.getMessage()}
        extra = getattr(record, "ctx", None)
        if extra:
            d.update(extra)
        return json.dumps(d)


def get_logger(name: str = "esd") -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(_JsonFormatter())
        log.addHandler(h)
        log.setLevel(logging.INFO)
    return log
