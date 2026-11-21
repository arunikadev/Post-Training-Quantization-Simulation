"""Unified console and file logging setup."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "ptq_sim",
    log_file: str | Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure and return a logger with clean console and optional UTF-8 file output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Reset existing handlers so repeated runs do not duplicate log lines.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def close_file_handlers(logger: logging.Logger) -> None:
    """Flush and close any FileHandlers attached to logger (prevents Windows file locks)."""
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            try:
                handler.flush()
                handler.close()
            finally:
                logger.removeHandler(handler)
