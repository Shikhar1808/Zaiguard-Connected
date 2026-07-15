"""
main.py

Entry point.

Usage:
    uv run main.py                    # preview window + alerts
    uv run main.py --no-preview       # headless (server / SSH)
    uv run main.py --log DEBUG
"""

from __future__ import annotations

import argparse
import sys

from core.config_loader import load_config
from core.logger import log, setup_logging
from core.pipeline import Pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Surveillance pipeline")
    p.add_argument("--config",     default="config", help="Config directory")
    p.add_argument("--log",        default="INFO",   help="Log level")
    p.add_argument("--no-preview", action="store_true", help="Disable preview window")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log)

    log.info("Loading config from: {}", args.config)
    try:
        config = load_config(args.config)
    except Exception as exc:
        log.error("Failed to load config: {}", exc)
        sys.exit(1)

    log.info(
        "Config loaded | cameras={} schedules={}",
        len(config.cameras), len(config.schedules),
    )

    pipeline = Pipeline(config)
    pipeline.start()

    if args.no_preview:
        pipeline.wait()
    else:
        pipeline.run_with_preview()   # blocks main thread (OpenCV requirement)

    pipeline.stop()


if __name__ == "__main__":
    main()