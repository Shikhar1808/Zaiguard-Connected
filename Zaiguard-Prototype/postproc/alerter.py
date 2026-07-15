"""
postproc/alerter.py

Final stage. Dispatches ConfirmedAlerts to all sinks.

Sinks
-----
  ✓  Loguru WARNING line
  ✓  Compact JSONL  →  outputs/alerts/alert_log.jsonl
  ✓  Full JSON      →  outputs/alerts/YYYY-MM-DD/<alert_id>.json

Future stubs: _dispatch_redis, _dispatch_timescale, _dispatch_qdrant
"""

from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path

from core.logger import log
from core.packets import ConfirmedAlert


class Alerter(threading.Thread):
    def __init__(
        self,
        in_queue: queue.Queue,
        stop_event: threading.Event,
        output_dir: str = "outputs",
        config=None,
    ) -> None:
        super().__init__(name="alerter", daemon=True)
        self.in_queue   = in_queue
        self.stop_event = stop_event
        self._alert_dir = Path(output_dir) / "alerts"
        self._alert_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl = self._alert_dir / "alert_log.jsonl"

        # Alert Engine integration — fire-and-forget HTTP client
        self._alert_engine_client = None
        if config:
            ae_cfg = config.extensions.get("alert_engine", {})
            if ae_cfg.get("enabled"):
                from postproc.alert_engine_client import AlertEngineClient
                self._alert_engine_client = AlertEngineClient(
                    base_url=ae_cfg.get("url", "http://localhost:8000"),
                    timeout=ae_cfg.get("timeout_s", 5),
                )

    def _dispatch_log(self, alert: ConfirmedAlert) -> None:
        v = alert.meta.threshold_verdict
        log.warning(
            "ALERT FIRED | id={}  type={:<15}  cam={}  track={}  "
            "conf={:.3f}  time={}  [{}/{}]  clip={}",
            alert.alert_id[:8],
            alert.event_type,
            alert.camera_id,
            alert.track_id,
            alert.confidence,
            alert.meta.wall_time,
            v.frames_in_violation,
            v.frames_evaluated,
            alert.clip_path or "none",
        )

    def _dispatch_jsonl(self, alert: ConfirmedAlert) -> None:
        compact = {
            "schema_version": alert.schema_version,
            "alert_id":   alert.alert_id,
            "ts_iso":     alert.ts_iso,
            "event_type": alert.event_type,
            "camera_id":  alert.camera_id,
            "camera_label": alert.meta.camera_label,
            "track_id":   alert.track_id,
            "confidence": alert.confidence,
            "severity":   alert.meta.severity,
            "wall_time":  alert.meta.wall_time,
            "hour":       alert.meta.hour_of_day,
            "day":        alert.meta.day_of_week,
            "schedule":   alert.meta.schedule_summary,
            "clip_path":      alert.clip_path,
            "snapshot_path":  alert.snapshot_path,
            "detection_score": alert.meta.detection_score,
            "bbox_norm":  alert.meta.bbox_norm,
            "frame_shape": [alert.meta.frame_height, alert.meta.frame_width],
            "verdict": {
                "score":  alert.meta.threshold_verdict.raw_score,
                "viol":   alert.meta.threshold_verdict.frames_in_violation,
                "of":     alert.meta.threshold_verdict.frames_evaluated,
            },
            "scene": {
                "persons": alert.meta.scene.person_count,
                "tracks":  alert.meta.scene.track_count,
                "crowd_density": alert.meta.scene.crowd_density,
            },
            "track_duration_s": alert.meta.track_duration_s,
        }
        # Part 2 extensibility fields — only include when non-empty
        if alert.pipeline_id:
            compact["pipeline_id"] = alert.pipeline_id
        if alert.extra:
            compact["extra"] = alert.extra
        try:
            with open(self._jsonl, "a") as f:
                f.write(json.dumps(compact) + "\n")
        except Exception as exc:
            log.warning("JSONL write failed: {}", exc)

    def _dispatch_full_json(self, alert: ConfirmedAlert) -> None:
        try:
            date_dir = self._alert_dir / time.strftime("%Y-%m-%d")
            date_dir.mkdir(parents=True, exist_ok=True)
            path = date_dir / f"{alert.alert_id}.json"
            path.write_text(alert.model_dump_json(indent=2))
        except Exception as exc:
            log.warning("Full JSON write failed: {}", exc)

    # Future stubs:
    # def _dispatch_redis(self, alert): ...
    # def _dispatch_timescale(self, alert): ...
    # def _dispatch_qdrant(self, alert): ...

    def _dispatch(self, alert: ConfirmedAlert) -> None:
        self._dispatch_log(alert)
        self._dispatch_jsonl(alert)
        self._dispatch_full_json(alert)
        # Send to Alert Engine (fire-and-forget, non-blocking)
        if self._alert_engine_client:
            self._alert_engine_client.send(alert)

    def run(self) -> None:
        log.info("Alerter started | output={}", self._alert_dir)
        while not self.stop_event.is_set():
            try:
                alert: ConfirmedAlert = self.in_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            self._dispatch(alert)
        log.info("Alerter stopped")