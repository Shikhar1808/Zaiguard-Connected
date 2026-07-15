"""
ingestion/frame_sampler.py

Single daemon thread — motion gate + rate limiter between raw_frames and sampled_frames.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import defaultdict
from typing import Optional

import cv2
import numpy as np

from core.config_loader import ThresholdConfig
from core.logger import log
from core.packets import FramePacket


class FrameSampler(threading.Thread):
    def __init__(
        self,
        in_queue: queue.Queue,
        out_queue: queue.Queue,
        thresholds: ThresholdConfig,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name="frame-sampler", daemon=True)
        self.in_queue    = in_queue
        self.out_queue   = out_queue
        self.thresholds  = thresholds
        self.stop_event  = stop_event
        self._prev_gray: dict[str, Optional[np.ndarray]] = defaultdict(lambda: None)
        self._last_sent: dict[str, float] = defaultdict(float)
        self._min_interval = 1.0 / max(thresholds.motion_sample_fps, 1)

    def _motion_score(self, camera_id: str, frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev = self._prev_gray[camera_id]
        self._prev_gray[camera_id] = gray
        if prev is None:
            return 255.0
        return float(cv2.absdiff(gray, prev).mean())

    def run(self) -> None:
        log.info(
            "FrameSampler started | motion_threshold={} max_fps={}",
            self.thresholds.motion_threshold, self.thresholds.motion_sample_fps,
        )
        total_in = total_out = 0

        while not self.stop_event.is_set():
            try:
                packet: FramePacket = self.in_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            total_in += 1
            cam_id = packet.camera_id
            now    = time.monotonic()

            if now - self._last_sent[cam_id] < self._min_interval:
                continue

            score = self._motion_score(cam_id, packet.frame)
            if score < self.thresholds.motion_threshold:
                log.debug("[{}] dropped — low motion ({:.2f})", cam_id, score)
                continue

            self._last_sent[cam_id] = now
            total_out += 1

            try:
                self.out_queue.put_nowait(packet)
            except queue.Full:
                log.debug("[{}] sampled_frames full — dropped", cam_id)

            if total_in % 500 == 0:
                log.info(
                    "Sampler | {} in / {} out ({:.1f}% pass)",
                    total_in, total_out, total_out / total_in * 100,
                )

        log.info("FrameSampler stopped | {} in / {} out", total_in, total_out)