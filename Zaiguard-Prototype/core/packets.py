"""
core/packets.py

Pydantic v2 models for every inter-stage payload.

Schema change from zone-based to camera-schedule-based
-------------------------------------------------------
AlertMeta no longer carries zone_polygon or distance_to_zone_edge_px.
ThresholdVerdict.zone_gate_passed is renamed to camera_restricted — it simply
records whether the camera was in a restricted state at the time of the alert.
Everything else is unchanged.

SCHEMA_VERSION must be bumped whenever a field is added or renamed.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.4.0"


# ─────────────────────────────────────────────────────────────────────────────
# Ingestion → Sampler → Backbone
# ─────────────────────────────────────────────────────────────────────────────

class FramePacket(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    camera_id: str
    frame_id: int
    timestamp: float        # unix epoch seconds
    frame: Any              # np.ndarray BGR uint8 H×W×3


# ─────────────────────────────────────────────────────────────────────────────
# Per-track — populated by backbone.py
# ─────────────────────────────────────────────────────────────────────────────

class TrackBox(BaseModel):
    track_id: int
    bbox: list[int]             # [x1, y1, x2, y2] absolute pixels
    bbox_norm: list[float]      # [x1, y1, x2, y2] normalised 0–1
    score: float
    class_id: int
    class_label: str
    centroid: list[int]         # [cx, cy] pixels
    centroid_norm: list[float]  # [cx, cy] normalised 0–1
    area_px: int
    aspect_ratio: float


# ─────────────────────────────────────────────────────────────────────────────
# Scene-level features — per frame
# ─────────────────────────────────────────────────────────────────────────────

class SceneFeatures(BaseModel):
    track_count: int
    person_count: int
    mean_speed_px_per_s: float = 0.0
    frame_motion_score: float = 0.0
    crowd_density: float = 0.0      # persons per 1M pixels


# ─────────────────────────────────────────────────────────────────────────────
# Backbone + Tracker → Classifiers
# ─────────────────────────────────────────────────────────────────────────────

class TrackPacket(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    camera_id: str
    frame_id: int
    timestamp: float
    frame: Any                  # np.ndarray — not serialised
    frame_shape: list[int]      # [H, W, C]
    tracks: list[TrackBox]
    scene_features: SceneFeatures


# ─────────────────────────────────────────────────────────────────────────────
# Threshold engine verdict
# ─────────────────────────────────────────────────────────────────────────────

class ThresholdVerdict(BaseModel):
    """Full audit record of one threshold evaluation."""
    window_size: int
    frames_in_violation: int
    frames_evaluated: int
    raw_score: float            # frames_in_violation / frames_evaluated
    threshold: float
    passed: bool
    cooldown_active: bool
    # Time gate result: was the current hour inside an allowed window?
    # False = restricted = violation condition met.
    camera_restricted: bool
    # Human-readable: which schedule window(s) apply to this camera
    schedule_summary: str


# ─────────────────────────────────────────────────────────────────────────────
# Alert metadata — JSON-safe, no numpy
# ─────────────────────────────────────────────────────────────────────────────

class AlertMeta(BaseModel):
    # ── spatial ───────────────────────────────────────────────────────────────
    bbox: list[int]
    bbox_norm: list[float]
    centroid: list[int]
    centroid_norm: list[float]
    area_px: int
    aspect_ratio: float

    # ── temporal / schedule ───────────────────────────────────────────────────
    # ISO wall-clock time at alert — human readable without needing timestamp math
    wall_time: str              # e.g. "02:17:43"
    hour_of_day: int            # 0–23
    day_of_week: str            # "Monday" … "Sunday"
    schedule_summary: str       # e.g. "allowed 08:00–18:00, 20:00–22:00"
    violation_history: list[bool]
    threshold_verdict: ThresholdVerdict

    # ── scene ─────────────────────────────────────────────────────────────────
    scene: SceneFeatures

    # ── track ─────────────────────────────────────────────────────────────────
    class_label: str
    detection_score: float

    # ── camera ────────────────────────────────────────────────────────────────
    camera_id: str
    camera_label: str

    # ── resolution (v1.3) — needed for bbox denormalization in dashboard ──────
    frame_width: int = 0
    frame_height: int = 0

    # ── presence duration (v1.3) — how long this track has been observed ──────
    track_duration_s: float = 0.0         # seconds since track first appeared
    track_first_seen_frame: int = 0       # frame_id when track first appeared

    # ── severity (v1.3) — auto-classified from score + context ────────────────
    severity: str = "medium"              # "low" | "medium" | "high" | "critical"


# ─────────────────────────────────────────────────────────────────────────────
# Classifiers → Deduplicator
# ─────────────────────────────────────────────────────────────────────────────

class AlertCandidate(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    schema_version: str = SCHEMA_VERSION
    camera_id: str
    frame_id: int
    timestamp: float
    event_type: str
    track_id: int
    confidence: float
    zone_id: str | None = None      # None for camera-level alerts; kept for future zone classifiers
    meta: AlertMeta
    extra: dict[str, Any] = Field(default_factory=dict)   # extensible bag for Part 2 classifiers
    _frame: Any = None              # internal — stripped before writing to disk


# ─────────────────────────────────────────────────────────────────────────────
# Deduplicator → Alerter → Storage
# ─────────────────────────────────────────────────────────────────────────────

class ConfirmedAlert(BaseModel):
    schema_version: str = SCHEMA_VERSION
    alert_id: str
    pipeline_id: str = ""            # identifies pipeline instance (for multi-node Part 2)
    camera_id: str
    timestamp: float
    ts_iso: str
    event_type: str
    track_id: int
    confidence: float
    zone_id: str | None = None
    clip_path: str | None = None
    snapshot_path: str | None = None      # annotated JPEG snapshot at moment of firing
    meta: AlertMeta
    extra: dict[str, Any] = Field(default_factory=dict)   # extensible bag for Part 2 classifiers