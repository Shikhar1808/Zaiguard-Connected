"""
postproc/storage_cleanup.py

Automatic retention-based cleanup of old clips and alert files.

Runs once on pipeline start, then periodically (every 6 hours).
Deletes date-subdirectories in outputs/clips/ and outputs/alerts/
older than `retention_days`.

Set retention_days=0 in config to disable entirely.
"""

from __future__ import annotations

import shutil
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

from core.logger import log

_CLEANUP_INTERVAL_S = 6 * 3600   # run every 6 hours


class StorageCleanup(threading.Thread):
    def __init__(
        self,
        stop_event: threading.Event,
        output_dir: str = "outputs",
        retention_days: int = 30,
    ) -> None:
        super().__init__(name="storage-cleanup", daemon=True)
        self.stop_event     = stop_event
        self._output_dir    = Path(output_dir)
        self._retention_days = retention_days

    def _cleanup_once(self) -> None:
        """Delete date-subdirectories older than retention_days."""
        if self._retention_days <= 0:
            return

        cutoff = datetime.now() - timedelta(days=self._retention_days)
        cutoff_str = cutoff.strftime("%Y-%m-%d")

        for subdir_name in ("clips", "alerts"):
            target = self._output_dir / subdir_name
            if not target.is_dir():
                continue
            for child in sorted(target.iterdir()):
                if not child.is_dir():
                    continue
                # Only process directories that look like date folders (YYYY-MM-DD)
                name = child.name
                if len(name) != 10 or name[4] != '-' or name[7] != '-':
                    continue
                try:
                    if name < cutoff_str:
                        item_count = sum(1 for _ in child.rglob("*"))
                        shutil.rmtree(child)
                        log.info(
                            "Cleanup | removed {} ({} items, older than {} days)",
                            child, item_count, self._retention_days,
                        )
                except Exception as exc:
                    log.warning("Cleanup failed for {}: {}", child, exc)

    def run(self) -> None:
        if self._retention_days <= 0:
            log.info("StorageCleanup disabled (retention_days=0)")
            return

        log.info(
            "StorageCleanup started | retention={}d interval={}h output={}",
            self._retention_days, _CLEANUP_INTERVAL_S // 3600, self._output_dir,
        )

        # Run immediately on startup
        self._cleanup_once()

        # Then periodically
        while not self.stop_event.is_set():
            # Sleep in small increments so we can respond to stop_event quickly
            slept = 0.0
            while slept < _CLEANUP_INTERVAL_S and not self.stop_event.is_set():
                time.sleep(min(30.0, _CLEANUP_INTERVAL_S - slept))
                slept += 30.0
            if not self.stop_event.is_set():
                self._cleanup_once()

        log.info("StorageCleanup stopped")
