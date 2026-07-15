"""
inference/classifier_runner.py

Single thread that owns all registered classifiers.
"""

from __future__ import annotations

import queue
import threading

from classifiers.base_classifier import BaseClassifier
from core.logger import log
from core.packets import TrackPacket

_PRUNE_EVERY_N = 100


class ClassifierRunner(threading.Thread):
    def __init__(
        self,
        in_queue: queue.Queue,
        out_queue: queue.Queue,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name="classifier-runner", daemon=True)
        self.in_queue    = in_queue
        self.out_queue   = out_queue
        self.stop_event  = stop_event
        self._classifiers: list[BaseClassifier] = []
        self._n = 0

    def register(self, clf: BaseClassifier) -> None:
        clf.setup()
        clf.ready.wait(timeout=10)
        self._classifiers.append(clf)
        log.info("Registered classifier: {}", clf.event_type)

    def run(self) -> None:
        log.info("ClassifierRunner started | {} classifier(s)", len(self._classifiers))

        while not self.stop_event.is_set():
            try:
                packet: TrackPacket = self.in_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            self._n += 1

            for clf in self._classifiers:
                try:
                    alerts = clf.classify(packet)
                except Exception as exc:
                    log.exception("Classifier {} raised: {}", clf.event_type, exc)
                    alerts = []

                for alert in alerts:
                    try:
                        self.out_queue.put_nowait(alert)
                    except queue.Full:
                        log.debug("alert_candidates full — dropped")

            if self._n % _PRUNE_EVERY_N == 0:
                active = {t.track_id for t in packet.tracks}
                for clf in self._classifiers:
                    if hasattr(clf, "prune_stale_tracks"):
                        clf.prune_stale_tracks(active, packet.camera_id)

        for clf in self._classifiers:
            clf.teardown()
        log.info("ClassifierRunner stopped")