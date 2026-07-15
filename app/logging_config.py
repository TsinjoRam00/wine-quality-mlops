from __future__ import annotations

import os
import sys
from logging.config import dictConfig


def configure_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {
                    "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                    "format": (
                        "%(asctime)s %(levelname)s %(name)s %(message)s "
                        "%(request_id)s"
                    ),
                }
            },
            "handlers": {
                "stdout": {
                    "class": "logging.StreamHandler",
                    "stream": sys.stdout,
                    "formatter": "json",
                }
            },
            "root": {"handlers": ["stdout"], "level": level},
        }
    )
