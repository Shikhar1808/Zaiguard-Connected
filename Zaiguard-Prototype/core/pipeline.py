"""
core/pipeline.py

Wires every pipeline stage. The only place threads are created and joined.

Multi-camera
------------
No hard upper limit on camera count — start_readers() spawns one
CameraReader thread per enabled camera, and PreviewRenderer creates one
window per enabled camera. A shared CameraStatus board lets the preview
show a NO INPUT placeholder for any camera whose source (RTSP / file /
HTTP / device fallback chain) could not be opened.

Minimum 1 camera is enforced at config-load time (core/config_loader.py).
"""

from __future__ import annotations

import threading
import time

from classifiers.unauth_access import UnauthAccessClassifier
from core.config_loader import AppConfig
from core.logger import log
from core.queue_bus import QueueBus
from inference.backbone import BackboneInference
from inference.classifier_runner import ClassifierRunner
from ingestion.frame_sampler import FrameSampler
from ingestion.rtsp_reader import CameraStatus, start_readers
from postproc.alerter import Alerter
from postproc.deduplicator import Deduplicator
from postproc.preview import PreviewRenderer
from postproc.storage_cleanup import StorageCleanup


class Pipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config       = config
        self.stop_event    = threading.Event()
        self.bus            = QueueBus(maxsize=config.queue_maxsize)
        self.status_board   = CameraStatus()
        self._threads: list[threading.Thread] = []
        self._readers: list[threading.Thread] = []
        self._preview: PreviewRenderer | None = None

    def _build(self) -> None:
        cfg = self.config
        ev  = self.stop_event
        bus = self.bus

        self.sampler = FrameSampler(
            in_queue=bus.raw_frames,
            out_queue=bus.sampled_frames,
            thresholds=cfg.thresholds,
            stop_event=ev,
        )
        self.backbone = BackboneInference(
            in_queue=bus.sampled_frames,
            out_queue=bus.tracks,
            thresholds=cfg.thresholds,
            stop_event=ev,
            preview_queue=bus.preview_frames,
        )
        self.runner = ClassifierRunner(
            in_queue=bus.tracks,
            out_queue=bus.alert_candidates,
            stop_event=ev,
        )
        self.runner.register(
            UnauthAccessClassifier(cfg.schedules, cfg.thresholds)
        )
        # self.runner.register(ViolenceClassifier(...))

        self.dedup = Deduplicator(
            in_queue=bus.alert_candidates,
            out_queue=bus.confirmed_alerts,
            stop_event=ev,
            thresholds=cfg.thresholds,
            output_dir=cfg.alert_output_dir,
            save_clips=cfg.save_clips,
            preview_alert_queue=bus.preview_alerts,
        )
        self.alerter = Alerter(
            in_queue=bus.confirmed_alerts,
            stop_event=ev,
            output_dir=cfg.alert_output_dir,
            config=cfg,
        )
        self.cleanup = StorageCleanup(
            stop_event=ev,
            output_dir=cfg.alert_output_dir,
            retention_days=cfg.retention_days,
        )

        self._threads = [
            self.sampler, self.backbone,
            self.runner, self.dedup, self.alerter,
            self.cleanup,
        ]

        self._preview = PreviewRenderer(
            config=cfg,
            preview_queue=bus.preview_frames,
            alert_queue=bus.preview_alerts,
            output_dir=cfg.alert_output_dir,
            status_board=self.status_board,
        )

    def start(self) -> None:
        log.info("Pipeline starting …")
        self._build()
        for t in self._threads:
            t.start()
        self._readers = start_readers(
            cameras=self.config.cameras,
            out_queue=self.bus.raw_frames,
            stop_event=self.stop_event,
            status_board=self.status_board,
        )
        log.info(
            "Pipeline running | cameras={} classifiers={}",
            len(self._readers), len(self.runner._classifiers),
        )

    def stop(self) -> None:
        log.info("Pipeline stopping …")
        self.stop_event.set()
        if self._preview:
            self._preview.close()
        for t in self._threads + self._readers:
            t.join(timeout=5)
        log.info("Pipeline stopped")

    def run_with_preview(self) -> None:
        """
        Blocks the MAIN thread running the OpenCV window loop.
        Must be called from main thread on Windows/macOS.
        """
        assert self._preview is not None
        log.info("Preview active — press Q or Esc to quit")
        try:
            while not self.stop_event.is_set():
                if not self._preview.update():
                    log.info("Preview window closed by user")
                    break
        except KeyboardInterrupt:
            log.info("KeyboardInterrupt — shutting down")

    def wait(self) -> None:
        """Headless mode — no preview window."""
        try:
            while not self.stop_event.is_set():
                time.sleep(0.5)
        except KeyboardInterrupt:
            log.info("KeyboardInterrupt — shutting down")