"""
ZaiGuard Alert Engine — Structured Logging Setup
================================================
Configures structlog to intercept and format both standard library logs
and structlog calls. Outputs color-coded logs for development and JSON
formatted logs for production.
"""

from __future__ import annotations

import logging
import sys
import structlog
from config.settings import settings


def setup_logging() -> None:
    # Clear any existing root handlers
    root_logger = logging.getLogger()
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)

    # Define processors shared between standard logging and structlog
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    # Configure structlog wrapper and factory
    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Setup formatter based on environment (JSON in prod, ConsoleRenderer in dev)
    if settings.is_production:
        formatter = structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared_processors,
            processor=structlog.processors.JSONRenderer(),
        )
    else:
        formatter = structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared_processors,
            processor=structlog.dev.ConsoleRenderer(colors=True),
        )

    # Set up console stdout handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # Set up file handler (always JSON structured for parsing and diagnostics)
    from pathlib import Path
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    file_formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processor=structlog.processors.JSONRenderer(),
    )
    from logging.handlers import TimedRotatingFileHandler
    file_handler = TimedRotatingFileHandler(
        filename=log_dir / "alert_engine.log",
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8",
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Set logger level from settings
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    root_logger.setLevel(log_level)

    # Silence verbose third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
