"""
core/queue_bus.py

Owns every inter-thread Queue in the pipeline.
"""

from __future__ import annotations

import queue
from dataclasses import dataclass, field


@dataclass
class QueueBus:
    maxsize: int = 32

    # Camera readers  →  frame sampler
    raw_frames: queue.Queue = field(init=False)
    # Frame sampler   →  backbone
    sampled_frames: queue.Queue = field(init=False)
    # Backbone        →  classifiers
    tracks: queue.Queue = field(init=False)
    # Backbone fan-out  →  preview renderer
    preview_frames: queue.Queue = field(init=False)
    # Classifiers     →  deduplicator
    alert_candidates: queue.Queue = field(init=False)
    # Deduplicator    →  alerter
    confirmed_alerts: queue.Queue = field(init=False)
    # Deduplicator fan-out  →  preview renderer (alert flash + clip)
    preview_alerts: queue.Queue = field(init=False)

    def __post_init__(self) -> None:
        self.raw_frames       = queue.Queue(maxsize=self.maxsize)
        self.sampled_frames   = queue.Queue(maxsize=self.maxsize)
        self.tracks           = queue.Queue(maxsize=self.maxsize)
        self.preview_frames   = queue.Queue(maxsize=4)
        self.alert_candidates = queue.Queue(maxsize=self.maxsize)
        self.confirmed_alerts = queue.Queue(maxsize=self.maxsize)
        self.preview_alerts   = queue.Queue(maxsize=16)