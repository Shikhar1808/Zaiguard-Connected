"""
core/logger.py

Loguru setup. Call setup_logging() once in main.py.
Every module does:
    from core.logger import log        # module-level default
    from loguru import logger          # or import directly — same sink

Loguru features we use
----------------------
- Coloured console output with level, module, line number
- Automatic file rotation  →  logs/surveillance_{date}.log
- Structured .bind() for per-camera context injection
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def setup_logging(level: str = "INFO", log_dir: str = "logs") -> None:
    logger.remove()  # remove default sink

    # ── Console ────────────────────────────────────────────────────────────────
    logger.add(
        sys.stdout,
        level=level.upper(),
        format=(
            "<green>{time:HH:mm:ss}</green> "
            "<level>{level: <8}</level> "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # ── Rotating file ──────────────────────────────────────────────────────────
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger.add(
        f"{log_dir}/surveillance_{{time:YYYY-MM-DD}}.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{line} — {message}",
        rotation="00:00",      # new file each day
        retention="14 days",
        compression="gz",
        backtrace=True,
        diagnose=False,        # no local vars in file (security)
    )

    logger.info("Logging initialised — console={} file={}/surveillance_*.log", level, log_dir)


# Module-level convenience — same as `from loguru import logger`
log = logger