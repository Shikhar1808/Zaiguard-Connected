"""
ingestion/rtsp_reader.py

One CameraReader thread per camera. Reconnects automatically on failure.

Status tracking
----------------
Each reader publishes its connection status to a shared dict so the
preview renderer can show a "NO INPUT" placeholder instead of a blank
or missing window when a camera's source (RTSP, file, HTTP, all
fallback tiers including device scan) cannot be opened.

Status values
-------------
  "connecting"  initial state, or actively retrying after a failure
  "connected"   frames are flowing normally
  "failed"      every fallback tier in resolve_source() was exhausted
                on the most recent attempt (will keep retrying)
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import cv2

from core.config_loader import CameraConfig
from core.logger import log
from core.packets import FramePacket
from ingestion.source_resolver import resolve_source

_RECONNECT_DELAY_S = 5.0


class CameraStatus:
    """Thread-safe shared status board: camera_id -> status string."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._status: dict[str, str] = {}

    def set(self, camera_id: str, status: str) -> None:
        with self._lock:
            self._status[camera_id] = status

    def get(self, camera_id: str) -> str:
        with self._lock:
            return self._status.get(camera_id, "connecting")

    def snapshot(self) -> dict[str, str]:
        with self._lock:
            return dict(self._status)


class CameraReader(threading.Thread):
    def __init__(
        self,
        config: CameraConfig,
        out_queue,
        stop_event: threading.Event,
        status_board: CameraStatus | None = None,
    ) -> None:
        super().__init__(name=f"reader-{config.camera_id}", daemon=True)
        self.config       = config
        self.out_queue    = out_queue
        self.stop_event   = stop_event
        self.status_board = status_board
        self._frame_id    = 0
        self._cap: Optional[cv2.VideoCapture] = None
        self._set_status("connecting")

    def _set_status(self, status: str) -> None:
        if self.status_board is not None:
            self.status_board.set(self.config.camera_id, status)

    def _open(self) -> bool:
        cap, src = resolve_source(self.config.source, self.config.camera_id)
        if cap is None:
            self._set_status("failed")
            return False
        self._cap = cap
        if self.config.fps_cap:
            self._cap.set(cv2.CAP_PROP_FPS, self.config.fps_cap)
        self._set_status("connected")
        return True

    def _release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def run(self) -> None:
        log.info("[{}] Reader thread started", self.config.camera_id)

        while not self.stop_event.is_set():
            if self._cap is None or not self._cap.isOpened():
                self._release()
                if not self._open():
                    log.warning(
                        "[{}] No source available — retrying in {:.0f}s",
                        self.config.camera_id, _RECONNECT_DELAY_S,
                    )
                    self.stop_event.wait(_RECONNECT_DELAY_S)
                    continue

            t0 = time.monotonic()
            ok, frame = self._cap.read()
            if not ok or frame is None:
                log.warning("[{}] Read failed — reconnecting", self.config.camera_id)
                self._set_status("failed")
                self._release()
                self.stop_event.wait(_RECONNECT_DELAY_S)
                continue

            self._frame_id += 1
            packet = FramePacket(
                camera_id=self.config.camera_id,
                frame_id=self._frame_id,
                timestamp=time.time(),
                frame=frame,
            )
            try:
                self.out_queue.put_nowait(packet)
            except Exception:
                pass  # queue full — drop frame intentionally

            elapsed = time.monotonic() - t0
            sleep_for = (1.0 / max(self.config.fps_cap, 1)) - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

        self._release()
        log.info("[{}] Reader thread stopped", self.config.camera_id)


def start_readers(
    cameras: list[CameraConfig],
    out_queue,
    stop_event: threading.Event,
    status_board: CameraStatus | None = None,
) -> list[CameraReader]:
    """
    Spawn one CameraReader per enabled camera. No upper limit on count —
    each camera gets its own thread and its own status entry.
    """
    readers = []
    for cam in cameras:
        if not cam.enabled:
            log.info("Camera {} disabled — skipping", cam.camera_id)
            continue
        r = CameraReader(cam, out_queue, stop_event, status_board=status_board)
        r.start()
        readers.append(r)
    return readers