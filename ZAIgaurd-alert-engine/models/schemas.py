"""
ZaiGuard Alert Engine — Data Schemas (Integration Contract)
===========================================================
This file is the single source of truth for every data shape
that crosses a module boundary.

FOR TEAMMATES
-------------
- Upstream (Aastha's tracking pipeline): produce RawDetectionEvent
- Downstream (Vaibhav's dashboard): consume DashboardAlert
- Feedback path (dashboard → this module): send OperatorFeedback

Every field is documented. If something is unclear, ask before
assuming — a schema mismatch is the #1 cause of integration bugs.

WHY PYDANTIC
------------
- Free JSON serialization/deserialization (.model_dump(), .model_validate())
- Field-level validation with clear error messages on bad input
- Auto-generated OpenAPI docs in FastAPI (teammates can browse the spec)
- Type hints that IDEs and linters can check statically
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


# ─────────────────────────────────────────────────────────────
# Enumerations
# Using Enum instead of raw strings prevents typos like
# "violance" or "CRITCAL" slipping through undetected.
# ─────────────────────────────────────────────────────────────

class Pipeline(str, Enum):
    """
    The five event classifiers upstream of this module.
    Values match the pipeline identifiers used throughout
    the system (camera layer, feature extraction, etc.).
    """
    FIRE        = "fire"
    VIOLENCE    = "violence"
    DOG_ATTACK  = "dog_attack"
    TRESPASSING = "trespassing"
    ACCIDENT      = "accident"
    UNAUTH_ACCESS = "unauth_access"


class AlertTier(str, Enum):
    """
    Severity levels assigned in Layer 5.
    CRITICAL: immediate response required (fire, high-confidence violence)
    HIGH:     prompt attention required
    MEDIUM:   monitor and review
    LOW:      informational, near-threshold events
    """
    CRITICAL = "CRITICAL"
    HIGH     = "HIGH"
    MEDIUM   = "MEDIUM"
    LOW      = "LOW"


class OperatorAction(str, Enum):
    """
    The two feedback actions an operator can take on a DashboardAlert.
    DISMISS: mark as not important — triggers suppression rule creation
    CONFIRM: mark as genuine event — logs as confirmed, no suppression
    """
    DISMISS = "dismiss"
    CONFIRM = "confirm"


# ─────────────────────────────────────────────────────────────
# RawDetectionEvent
# ─────────────────────────────────────────────────────────────

class RawDetectionEvent(BaseModel):
    """
    INPUT to the Alert Engine.

    Produced by upstream classifiers (violence, fire, etc.) after
    the spatial/temporal feature extraction + tracking pipeline.
    This is what Aastha's module should produce and hand off.

    Nothing in this schema is specific to any single pipeline — all
    five classifiers produce this same shape, with pipeline_features
    carrying any pipeline-specific extras.
    """

    # Which classifier produced this event
    pipeline: Pipeline

    # Raw model confidence score, 0.0–1.0
    # The Alert Engine's threshold gate compares this against the
    # effective threshold computed from base + time + zone multipliers.
    raw_confidence: float = Field(..., ge=0.0, le=1.0)

    # Camera that captured the event
    camera_id: str = Field(..., min_length=1)

    # Zone identifier — must exist in zone_config table
    # (or "default" if zone data is unavailable)
    zone_id: str = Field(default="default")

    # Human-readable zone label (e.g., "gym_east", "parking_lot_north")
    # Passed through for display — the Alert Engine doesn't interpret this.
    zone_label: str = Field(default="Unknown Zone")

    # UTC timestamp of the detection
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Path or reference to the evidence clip buffer.
    # Owned by the recording pipeline — the Alert Engine only stores
    # this reference and passes it to the dashboard.
    # None if clip buffering is not available for this event.
    frame_ref: str | None = None

    # Tracked object IDs involved in the event.
    # These are ByteTrack stable IDs from the multi-object tracker.
    # Empty list = object IDs were not available at classification time.
    involved_ids: list[int] = Field(default_factory=list)

    # Pipeline-specific feature data.
    # Violence: {"inter_person_distance": float, "relative_velocity": float, ...}
    # Fire: {"smoke_region_area": float, "persistence_frames": int, ...}
    # Dog attack: {"dog_id": int, "human_id": int, "proximity": float, ...}
    # Trespassing: {"zone_rules_violated": list[str], ...}
    # Accident: {"conflict_score": float, "trajectory_overlap": float, ...}
    pipeline_features: dict[str, Any] = Field(default_factory=dict)

    @field_validator("timestamp")
    @classmethod
    def ensure_utc(cls, v: datetime) -> datetime:
        """Force all timestamps to UTC. Prevents timezone confusion."""
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)


# ─────────────────────────────────────────────────────────────
# AlertEvent
# ─────────────────────────────────────────────────────────────

class AlertEvent(BaseModel):
    """
    INTERNAL object produced by Layer 2 (Enrichment).

    This is what flows through Layers 3, 4A, 4B, and 5.
    It extends RawDetectionEvent with:
    - alert_id: deterministic UUID for idempotency
    - effective_conf: confidence adjusted by threshold context
    - hour_of_day / day_of_week: pre-computed for fast access downstream

    Teammates building other layers only need to understand this
    shape — they don't need to know about RawDetectionEvent.
    """

    # Deterministic ID derived from (camera_id, timestamp, pipeline).
    # Same underlying detection always produces the same alert_id,
    # so duplicate deliveries from the message bus are idempotent.
    # See models/schemas.py:build_alert_id() for the hash logic.
    alert_id: str

    # Original detection event, preserved in full for audit purposes
    source_event: RawDetectionEvent

    # Confidence after threshold context has been applied.
    # effective_conf = raw_confidence (the multipliers affect the
    # *threshold* you compare against, not the confidence itself).
    # Stored separately for clarity and dashboard display.
    effective_conf: float = Field(..., ge=0.0, le=1.0)

    # Pre-computed time fields — avoid recomputing in every layer
    hour_of_day: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)   # 0=Monday, 6=Sunday

    # Convenience accessors that delegate to source_event
    @property
    def pipeline(self) -> Pipeline:
        return self.source_event.pipeline

    @property
    def camera_id(self) -> str:
        return self.source_event.camera_id

    @property
    def zone_id(self) -> str:
        return self.source_event.zone_id

    @property
    def zone_label(self) -> str:
        return self.source_event.zone_label

    @property
    def timestamp(self) -> datetime:
        return self.source_event.timestamp

    @property
    def raw_confidence(self) -> float:
        return self.source_event.raw_confidence

    @property
    def frame_ref(self) -> str | None:
        return self.source_event.frame_ref

    @property
    def involved_ids(self) -> list[int]:
        return self.source_event.involved_ids

    @property
    def pipeline_features(self) -> dict[str, Any]:
        return self.source_event.pipeline_features


# ─────────────────────────────────────────────────────────────
# DashboardAlert
# ─────────────────────────────────────────────────────────────

class DashboardAlert(BaseModel):
    """
    OUTPUT of the Alert Engine.

    Everything the dashboard needs to display an alert, request
    acknowledgment, and submit operator feedback.

    suppression_score deserves explanation:
    Even when an alert is NOT suppressed, this field contains the
    cosine similarity of the nearest dismissed alert in Qdrant
    (0.0 if no similar dismissed alerts exist). This gives operators
    context like "this is 84% similar to something you dismissed before"
    without auto-hiding it — surfacing tuning opportunities.
    """

    alert_id: str
    tier: AlertTier
    pipeline: Pipeline
    camera_id: str
    zone_id: str
    zone_label: str
    raw_confidence: float
    effective_conf: float
    timestamp: datetime
    evidence_frame_ref: str | None
    involved_ids: list[int]

    # Cosine similarity against nearest dismissed alert in Qdrant.
    # 0.0 = no similar dismissed alerts found.
    # Values close to (but below) the suppression threshold indicate
    # a pattern worth investigating for suppression rule creation.
    suppression_score: float = Field(default=0.0, ge=0.0, le=1.0)


# ─────────────────────────────────────────────────────────────
# OperatorFeedback
# ─────────────────────────────────────────────────────────────

class OperatorFeedback(BaseModel):
    """
    FEEDBACK input from the dashboard operator.

    Sent to the /feedback endpoint when an operator acts on an alert.
    Triggers the suppression write-back path (Postgres + Qdrant).

    permanent only applies when action == "dismiss":
    - permanent=False (default): suppression rule expires in 24 hours
    - permanent=True: suppression rule has no expiry (operator is sure)
    """

    alert_id: str
    action: OperatorAction
    permanent: bool = False

    @model_validator(mode="after")
    def permanent_only_on_dismiss(self) -> "OperatorFeedback":
        """
        permanent=True on a CONFIRM action is a client bug.
        Catch it early with a clear error rather than silently ignoring.
        """
        if self.permanent and self.action != OperatorAction.DISMISS:
            raise ValueError(
                "'permanent' flag is only meaningful when action is 'dismiss'"
            )
        return self


# ─────────────────────────────────────────────────────────────
# Suppression rule (internal, used by Layer 4A and feedback handler)
# ─────────────────────────────────────────────────────────────

class SuppressionRule(BaseModel):
    """
    Represents a row in the suppression_rules table.
    Used internally by the exact rule store (Layer 4A) and
    the feedback handler when creating new rules.
    """
    camera_id: str
    zone_id: str | None = None
    pipeline: Pipeline
    hour_start: int | None = Field(default=None, ge=0, le=23)
    hour_end:   int | None = Field(default=None, ge=0, le=23)
    days_mask:  int | None = None       # bitmask, see §4.9 of architecture doc
    expires_at: datetime | None = None  # None = permanent
    source: str = "manual"

    @model_validator(mode="after")
    def validate_hour_range(self) -> "SuppressionRule":
        if (self.hour_start is None) != (self.hour_end is None):
            raise ValueError(
                "hour_start and hour_end must both be set or both be None"
            )
        if (
            self.hour_start is not None
            and self.hour_end is not None
            and self.hour_start > self.hour_end
        ):
            raise ValueError("hour_start must be <= hour_end")
        return self


# ─────────────────────────────────────────────────────────────
# Helper: deterministic alert_id generation
# ─────────────────────────────────────────────────────────────

def build_alert_id(camera_id: str, timestamp: datetime, pipeline: Pipeline) -> str:
    """
    Generates a deterministic, collision-resistant alert ID.

    WHY deterministic (not random UUID):
    The message bus (Redis Streams) guarantees at-least-once delivery,
    meaning the same detection event can arrive twice under failure
    conditions. A random UUID would create two separate alert records
    for the same real event. A deterministic ID means the second
    delivery produces the same ID, and the UNIQUE constraint on
    alert_log.alert_id silently rejects the duplicate — correct
    behavior with zero extra logic. (See §4.10 Idempotency)

    WHY SHA-256 truncated to 32 hex chars:
    SHA-256 is collision-resistant at any prefix length. 32 hex chars
    = 128 bits of entropy — astronomically unlikely to collide across
    all alerts ever generated. Shorter than a full UUID string but
    just as safe for this use case.
    """
    raw = f"{camera_id}|{timestamp.isoformat()}|{pipeline.value}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]