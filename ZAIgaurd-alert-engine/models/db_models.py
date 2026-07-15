"""
ZaiGuard Alert Engine — SQLAlchemy ORM Models
================================================
Python mirror of db/init/schemas.sql.

WHY THIS FILE EXISTS SEPARATELY FROM schemas.sql
--------------------------------------------------
schemas.sql is the source of truth for the actual database structure
(it's what Docker runs on first startup). This file lets the rest of
the Python codebase query and write to those same tables using
SQLAlchemy's ORM instead of hand-written SQL strings — type safety,
autocomplete, and protection against SQL injection for free.

If you change a column in schemas.sql, mirror the change here too.
The two files describe the same database from two different angles.

NAMING NOTE
-----------
SQLAlchemy reserves the name `metadata` on declarative models, so
table classes here are suffixed `Row` (e.g. ThresholdConfigRow) to
keep them visually distinct from the Pydantic schemas in
models/schemas.py, which represent the same concepts but for a
different purpose (API/data-contract validation, not DB storage).
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    ARRAY,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Shared base class for all ORM models in this module."""
    pass


# ─────────────────────────────────────────────────────────────
# threshold_config
# ─────────────────────────────────────────────────────────────

class ThresholdConfigRow(Base):
    """
    Mirrors: threshold_config table.
    One row per pipeline. Loaded once at startup and cached
    in memory by config/thresholds.py — not queried on every event.
    """
    __tablename__ = "threshold_config"

    pipeline: Mapped[str] = mapped_column(Text, primary_key=True)
    base_threshold: Mapped[float] = mapped_column(Float, nullable=False)
    dedup_ttl_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    escalation_delta: Mapped[float] = mapped_column(Float, nullable=False)


# ─────────────────────────────────────────────────────────────
# time_multiplier_config
# ─────────────────────────────────────────────────────────────

class TimeMultiplierConfigRow(Base):
    """Mirrors: time_multiplier_config table."""
    __tablename__ = "time_multiplier_config"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    hour_start: Mapped[int] = mapped_column(Integer, nullable=False)
    hour_end: Mapped[int] = mapped_column(Integer, nullable=False)
    multiplier: Mapped[float] = mapped_column(Float, nullable=False)
    label: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        CheckConstraint("hour_start >= 0 AND hour_end <= 23", name="valid_hours"),
    )


# ─────────────────────────────────────────────────────────────
# zone_config
# ─────────────────────────────────────────────────────────────

class ZoneConfigRow(Base):
    """Mirrors: zone_config table."""
    __tablename__ = "zone_config"

    zone_id: Mapped[str] = mapped_column(Text, primary_key=True)
    zone_label: Mapped[str] = mapped_column(Text, nullable=False)
    risk_multiplier: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)


# ─────────────────────────────────────────────────────────────
# tier_config
# ─────────────────────────────────────────────────────────────

class TierConfigRow(Base):
    """
    Mirrors: tier_config table.
    Multiple rows per pipeline. Application logic picks the
    first row (ordered by min_confidence descending) where
    effective_conf >= min_confidence.
    """
    __tablename__ = "tier_config"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pipeline: Mapped[str] = mapped_column(Text, nullable=False)
    min_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    tier: Mapped[str] = mapped_column(Text, nullable=False)


# ─────────────────────────────────────────────────────────────
# suppression_similarity_config
# ─────────────────────────────────────────────────────────────

class SuppressionSimilarityConfigRow(Base):
    """Mirrors: suppression_similarity_config table."""
    __tablename__ = "suppression_similarity_config"

    pipeline: Mapped[str] = mapped_column(Text, primary_key=True)
    similarity_threshold: Mapped[float] = mapped_column(Float, nullable=False)


# ─────────────────────────────────────────────────────────────
# suppression_rules
# ─────────────────────────────────────────────────────────────

class SuppressionRuleRow(Base):
    """
    Mirrors: suppression_rules table.
    Layer 4A (exact rule store) reads from this table.
    The feedback handler (Step 10) writes new rows here.
    """
    __tablename__ = "suppression_rules"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    camera_id: Mapped[str] = mapped_column(Text, nullable=False)
    zone_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    pipeline: Mapped[str] = mapped_column(Text, nullable=False)
    hour_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    hour_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    days_mask: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    source: Mapped[str] = mapped_column(Text, nullable=False, default="manual")

    __table_args__ = (
        CheckConstraint("source IN ('manual', 'auto_promoted')", name="valid_source"),
        # Mirrors idx_suppression_camera_pipeline from schemas.sql
        Index("idx_suppression_camera_pipeline", "camera_id", "pipeline"),
    )


# ─────────────────────────────────────────────────────────────
# alert_log
# ─────────────────────────────────────────────────────────────

class AlertLogRow(Base):
    """
    Mirrors: alert_log table (TimescaleDB hypertable, partitioned
    by `timestamp`). Permanent audit trail of every alert that
    reached the dashboard. alert_id is the deterministic ID from
    models/schemas.py:build_alert_id() — the UNIQUE/PRIMARY KEY
    constraint here is what makes duplicate event delivery harmless
    (see §4.10 Idempotency in the architecture document).
    """
    __tablename__ = "alert_log"

    alert_id: Mapped[str] = mapped_column(Text, primary_key=True)
    pipeline: Mapped[str] = mapped_column(Text, nullable=False)
    tier: Mapped[str] = mapped_column(Text, nullable=False)
    camera_id: Mapped[str] = mapped_column(Text, nullable=False)
    zone_id: Mapped[str] = mapped_column(Text, nullable=False)
    zone_label: Mapped[str] = mapped_column(Text, nullable=False)
    raw_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    effective_conf: Mapped[float] = mapped_column(Float, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True, nullable=False)
    evidence_frame_ref: Mapped[str | None] = mapped_column(Text, nullable=True)
    involved_ids: Mapped[list[int] | None] = mapped_column(ARRAY(Integer), nullable=True)
    suppression_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    operator_action: Mapped[str | None] = mapped_column(Text, nullable=True)
    action_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


# ─────────────────────────────────────────────────────────────
# outbox
# ─────────────────────────────────────────────────────────────

class OutboxRow(Base):
    """
    Mirrors: outbox table.
    Implements the Outbox Pattern (§4.1 of architecture document)
    to solve the dual-write problem between Postgres and Qdrant.

    Written in the SAME transaction as a suppression_rules insert.
    A background worker (scripts/auto_promote.py or a dedicated
    outbox worker, Step 10) reads unprocessed rows and performs
    the corresponding Qdrant write, retrying on failure.
    """
    __tablename__ = "outbox"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    event_type: Mapped[str] = mapped_column(Text, nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    failed_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index(
            "idx_outbox_unprocessed",
            "created_at",
            postgresql_where=(processed_at.is_(None)),  # type: ignore[name-defined]
        ),
    )