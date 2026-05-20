import sys
import json
import logging
import logging.config

from uuid import UUID
from pathlib import Path
from datetime import datetime

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        standard_attrs = logging.LogRecord(
            name="",
            level=0,
            pathname="",
            lineno=0,
            msg="",
            args=(),
            exc_info=None,
        ).__dict__.keys()

        for key, value in record.__dict__.items():
            if key not in standard_attrs:
                log_record[key] = value

        return json.dumps(
            log_record,
            ensure_ascii=False,
            default=self._json_default,
        )

    @staticmethod
    def _json_default(obj):
        if isinstance(obj, UUID):
            return str(obj)

        if isinstance(obj, Path):
            return str(obj)

        if isinstance(obj, datetime):
            return obj.isoformat()

        return str(obj)
    

JSON_LOG_FORMAT = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": JsonFormatter,
        },
    },
    "handlers": {
        "default": {
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
            "formatter": "json",
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["default"],
    },
}


def setup_logging(
    level: str | None = None,
) -> None:

    log_level = level or "DEBUG"

    logging.config.dictConfig(JSON_LOG_FORMAT)

    logging.getLogger().setLevel(log_level)

    logging.getLogger("uvicorn").setLevel(log_level)
    logging.getLogger("uvicorn.error").setLevel(log_level)
    logging.getLogger("uvicorn.access").setLevel("WARNING")

    logging.getLogger("sqlalchemy.engine").setLevel("WARNING")
    logging.getLogger("httpx").setLevel("WARNING")


def get_logger(name: str) -> logging.Logger:

    return logging.getLogger(name)


