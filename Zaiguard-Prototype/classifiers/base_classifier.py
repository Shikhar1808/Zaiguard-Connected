"""
classifiers/base_classifier.py

Abstract base class for all event classifiers.

Every classifier — unauth_access, violence, dog_attack, road_accident — must
subclass BaseClassifier and implement classify().

The pipeline calls classify() for every TrackPacket; classifiers return a
(possibly empty) list of AlertCandidates.

Design rules
------------
- Classifiers are STATEFUL: they maintain per-track history internally.
- Classifiers run in a single dedicated thread; they must not block indefinitely.
- Classifiers must not import GPU/model code at module level — load in setup().
- Classifiers signal readiness via the ready Event (pipeline waits before routing).
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod

from core.packets import AlertCandidate, TrackPacket


class BaseClassifier(ABC):
    """
    Subclass and implement:
        setup()     — called once before the pipeline starts
        classify()  — called for every TrackPacket
        teardown()  — called once on shutdown
    """

    #: Unique string identifier. Must match the key in thresholds.yaml.
    event_type: str = "base"

    def __init__(self) -> None:
        self.ready = threading.Event()
        self._stop = threading.Event()

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def setup(self) -> None:
        """
        One-time initialisation (load models, parse zone config, etc.).
        Call self.ready.set() when the classifier is ready to accept packets.
        Default implementation sets ready immediately (for logic-only classifiers).
        """
        self.ready.set()

    def teardown(self) -> None:
        """Release any resources (GPU memory, file handles, etc.)."""
        pass

    def stop(self) -> None:
        self._stop.set()

    # ── main interface ────────────────────────────────────────────────────────

    @abstractmethod
    def classify(self, packet: TrackPacket) -> list[AlertCandidate]:
        """
        Given a TrackPacket (one frame worth of tracks from the backbone),
        return a list of AlertCandidates that cross this classifier's threshold.

        Return an empty list if nothing is flagged.
        """
        ...

    # ── helpers subclasses commonly need ─────────────────────────────────────

    @staticmethod
    def bbox_centroid(bbox: list[int]) -> list[int]:
        x1, y1, x2, y2 = bbox
        return [(x1 + x2) // 2, (y1 + y2) // 2]

    @staticmethod
    def iou(a: list[int], b: list[int]) -> float:
        """Intersection-over-union of two [x1,y1,x2,y2] boxes."""
        ix1 = max(a[0], b[0])
        iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2])
        iy2 = min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0
