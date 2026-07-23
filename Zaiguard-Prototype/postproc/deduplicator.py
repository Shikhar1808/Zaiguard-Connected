"""
postproc/deduplicator.py

Two-gate filter + cross-camera deduplication.
Pushes ConfirmedAlerts to TWO queues:
  - confirmed_alerts  →  Alerter (JSONL + JSON writing)
  - preview_alerts    →  PreviewRenderer (flash + clip save)
Both are non-blocking puts; full queues drop silently.
"""

from __future__ import annotations

import queue
import threading
import time
import uuid
from collections import defaultdict
from pathlib import Path

from core.config_loader import ThresholdConfig
from core.logger import log
from core.packets import AlertCandidate, ConfirmedAlert


class Deduplicator(threading.Thread):
    def __init__(
        self,
        in_queue: queue.Queue,
        out_queue: queue.Queue,          # → Alerter
        stop_event: threading.Event,
        thresholds: ThresholdConfig,
        output_dir: str = "outputs",
        save_clips: bool = True,
        preview_alert_queue: queue.Queue | None = None,  # → PreviewRenderer
    ) -> None:
        super().__init__(name="deduplicator", daemon=True)
        self.in_queue            = in_queue
        self.out_queue           = out_queue
        self.preview_alert_queue = preview_alert_queue
        self.stop_event          = stop_event
        self.cfg                 = thresholds
        self.save_clips          = save_clips

        # (event_type, camera_id) → last confirmed timestamp
        self._last_global: dict[tuple, float] = defaultdict(float)

    def _min_conf(self, event_type: str) -> float:
        return self.cfg.unauth_min_confidence

    def _global_cooldown(self, event_type: str) -> float:
        return self.cfg.unauth_global_cooldown_s

    def _global_key(self, c: AlertCandidate) -> tuple:
        return (c.event_type, c.camera_id)

    def _in_global_cooldown(self, c: AlertCandidate) -> bool:
        elapsed = time.time() - self._last_global[self._global_key(c)]
        return elapsed < self._global_cooldown(c.event_type)

    def _touch(self, c: AlertCandidate) -> None:
        self._last_global[self._global_key(c)] = time.time()

    def run(self) -> None:
        log.info("Deduplicator started")

        while not self.stop_event.is_set():
            try:
                c: AlertCandidate = self.in_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # Gate 1 — minimum confidence
            if c.confidence < self._min_conf(c.event_type):
                log.debug("Dropped: conf {:.3f} < min {:.3f}",
                          c.confidence, self._min_conf(c.event_type))
                continue

            # Gate 2 — global per-camera cooldown
            if self._in_global_cooldown(c):
                log.debug("Suppressed (cooldown): {} cam={}", c.event_type, c.camera_id)
                continue

            self._touch(c)
            alert_id = str(uuid.uuid4())

            confirmed = ConfirmedAlert(
                alert_id=alert_id,
                camera_id=c.camera_id,
                timestamp=c.timestamp,
                ts_iso=time.strftime(
                    "%Y-%m-%dT%H:%M:%S", time.localtime(c.timestamp)
                ),
                event_type=c.event_type,
                track_id=c.track_id,
                confidence=c.confidence,
                zone_id=c.zone_id,
                clip_path=None,       # set by PreviewRenderer after saving
                snapshot_path=None,   # set by PreviewRenderer after saving
                embeddings=c.embeddings,
                meta=c.meta,
                extra=c.extra,
            )

            log.warning(
                "CONFIRMED | id={}  cam={}  track={}  time={}  conf={:.3f}  severity={}",
                alert_id[:8], confirmed.camera_id,
                confirmed.track_id, confirmed.meta.wall_time,
                confirmed.confidence, confirmed.meta.severity,
            )

            # Push to alerter (JSONL / JSON writing)
            try:
                self.out_queue.put_nowait(confirmed)
            except queue.Full:
                log.debug("confirmed_alerts full — dropped")

            # Push to preview renderer (flash + clip save)
            if self.preview_alert_queue is not None:
                try:
                    self.preview_alert_queue.put_nowait(confirmed)
                except queue.Full:
                    pass

        log.info("Deduplicator stopped")