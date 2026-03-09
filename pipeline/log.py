"""
Shared loguru configuration for the where-to-live pipeline.

Pipeline files just do `from loguru import logger` and use it directly.
The sinks (console + optional log file) are configured once by calling setup()
from the entry point (run_pipeline.py or standalone scripts).
"""

import sys
from pathlib import Path

from loguru import logger


def setup(log_file: Path | str | None = None, level: str = "DEBUG") -> None:
    """
    Configure loguru sinks. Call once at startup.

    Args:
        log_file: path to write persistent log file (optional)
        level: minimum log level to capture
    """
    logger.remove()  # drop loguru's default stderr handler

    # Console — colored, concise, human-readable
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan> | "
            "{message}"
        ),
        colorize=True,
        level=level,
    )

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_file),
            rotation="10 MB",
            retention="30 days",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            level=level,
            encoding="utf-8",
        )
        logger.debug(f"Logging to {log_file}")
