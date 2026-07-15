"""
postproc/alert_engine_client.py

HTTP client that sends ConfirmedAlerts to the ZaiGuard Alert Engine.

Maps the Prototype's ConfirmedAlert schema to the Alert Engine's
RawDetectionEvent schema and POSTs to /events. Fire-and-forget:
failures are logged but never crash the pipeline.

The Alert Engine generates its own 384-d sentence embeddings for
semantic suppression — we only need to send metadata fields, not
the Prototype's raw embedding vectors.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone

import requests

from core.logger import log
from core.packets import ConfirmedAlert


class AlertEngineClient:
    """
    Sends ConfirmedAlerts to the ZaiGuard Alert Engine's POST /events endpoint.

    All sends happen in a background thread so the Alerter's dispatch loop
    is never blocked by network I/O. Failed sends are logged and dropped —
    the pipeline keeps running regardless.
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 5) -> None:
        self._url = f"{base_url.rstrip('/')}/events"
        self._timeout = timeout
        log.info("AlertEngineClient initialized | url={} timeout={}s", self._url, timeout)

    def _map_to_raw_event(self, alert: ConfirmedAlert) -> dict:
        """
        Maps a ConfirmedAlert to the Alert Engine's RawDetectionEvent JSON.

        Only sends metadata fields that the Alert Engine actually uses:
        - build_alert_description() uses: pipeline, camera_id, zone_id,
          zone_label, timestamp, pipeline_features
        - Layer 1 threshold gate uses: raw_confidence
        - Layer 5 tiering uses: effective confidence

        The Alert Engine generates its own 384-d sentence embeddings from
        the text description — the Prototype's 128-d/6-d/4-d embeddings
        are NOT needed.
        """
        # Convert unix timestamp to ISO 8601 UTC datetime
        ts = datetime.fromtimestamp(alert.timestamp, tz=timezone.utc).isoformat()

        # Build pipeline_features with only the metadata the Alert Engine
        # needs for description building and dashboard display
        pipeline_features = {
            "severity": alert.meta.severity,
            "detection_score": alert.meta.detection_score,
            "wall_time": alert.meta.wall_time,
            "schedule_summary": alert.meta.schedule_summary,
            "person_count": alert.meta.scene.person_count,
            "track_count": alert.meta.scene.track_count,
            "bbox_norm": alert.meta.bbox_norm,
            "area_px": alert.meta.area_px,
            "track_duration_s": alert.meta.track_duration_s,
            "frame_shape": [alert.meta.frame_height, alert.meta.frame_width],
        }

        return {
            "pipeline": alert.event_type,
            "raw_confidence": alert.confidence,
            "camera_id": alert.camera_id,
            "zone_id": alert.zone_id or alert.camera_id,
            "zone_label": alert.meta.camera_label,
            "timestamp": ts,
            "frame_ref": alert.snapshot_path,
            "involved_ids": [alert.track_id],
            "pipeline_features": pipeline_features,
        }

    def _send_sync(self, alert: ConfirmedAlert) -> None:
        """Blocking send — runs in a background thread."""
        try:
            payload = self._map_to_raw_event(alert)
            resp = requests.post(
                self._url,
                json=payload,
                timeout=self._timeout,
            )

            if resp.status_code == 200:
                data = resp.json()
                if data.get("alert"):
                    log.info(
                        "Alert Engine accepted | id={} tier={}",
                        data["alert"].get("alert_id", "?")[:8],
                        data["alert"].get("tier", "?"),
                    )
                else:
                    log.debug(
                        "Alert Engine filtered (event dropped by pipeline) | cam={} type={}",
                        alert.camera_id, alert.event_type,
                    )
            else:
                log.warning(
                    "Alert Engine returned HTTP {} | cam={} type={}",
                    resp.status_code, alert.camera_id, alert.event_type,
                )

        except requests.ConnectionError:
            log.warning(
                "Alert Engine unreachable at {} — local files still saved",
                self._url,
            )
        except requests.Timeout:
            log.warning(
                "Alert Engine timeout ({}s) — local files still saved",
                self._timeout,
            )
        except Exception as exc:
            log.warning("Alert Engine send failed: {} — local files still saved", exc)

    def send(self, alert: ConfirmedAlert) -> None:
        """
        Fire-and-forget: sends alert to Alert Engine in a background thread.
        Never blocks the Alerter's dispatch loop, never crashes the pipeline.
        """
        t = threading.Thread(
            target=self._send_sync,
            args=(alert,),
            daemon=True,
            name="alert-engine-send",
        )
        t.start()
