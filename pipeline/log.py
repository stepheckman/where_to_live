"""
Shared logging configuration for the where-to-live pipeline.

Pipeline files use `from pipeline.log import logger` or just
`import logging; logger = logging.getLogger(__name__)`.

The sinks (console + optional log file) are configured once by calling setup()
from the entry point (run_pipeline.py or standalone scripts).
"""

import logging
import sys
from pathlib import Path

logger = logging.getLogger("pipeline")


def setup(log_file: Path | str | None = None, level: str = "DEBUG") -> None:
    """
    Configure logging handlers. Call once at startup.

    Args:
        log_file: path to write persistent log file (optional)
        level: minimum log level to capture
    """
    root = logging.getLogger("pipeline")
    root.setLevel(getattr(logging, level.upper(), logging.DEBUG))

    # Console — concise, human-readable
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(getattr(logging, level.upper(), logging.DEBUG))
    console.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    root.addHandler(console)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        from logging.handlers import RotatingFileHandler
        fh = RotatingFileHandler(
            str(log_file),
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        fh.setLevel(getattr(logging, level.upper(), logging.DEBUG))
        fh.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s",
        ))
        root.addHandler(fh)
        root.debug("Logging to %s", log_file)
