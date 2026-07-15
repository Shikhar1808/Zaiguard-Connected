"""
ZaiGuard Alert Engine — Pipeline Orchestrator
================================================
The single entry point that wires Layers 1–5 together.

Takes one RawDetectionEvent. Returns either:
  - None             → the event was dropped at some layer
  - DashboardAlert    → the event survived all five layers

THE FLOW
--------
    RawDetectionEvent
            |
            v
    Layer 1: threshold_gate.run_threshold_gate()
            | (fail -> log + return None)
            v
    Layer 2: enrichment.run_enrichment()
            v
    Layer 3: dedup.run_dedup()                       [Redis]
            | (duplicate -> log + return None)
            v
    Layer 4A: exact_rules.run_exact_rule_check()      [Postgres]
            | (suppressed -> log + return None)
            v
    Layer 4B: semantic.run_semantic_suppression_check() [Qdrant]
            | (suppressed -> log + return None)
            v
    Layer 5: tiering.run_tiering() + build_dashboard_alert()
            v
    DashboardAlert  ->  written to alert_log (Postgres)
                    ->  returned to caller (main.py pushes to dashboard)

WHY ONE FUNCTION OWNS THE WHOLE SEQUENCE
------------------------------------------
Each layer is independently testable in isolation (see tests/test_*.py
for Layers 1-5). This file is the only place that knows the ORDER they
run in and what happens when one of them says "stop." No other file
should re-implement this sequencing — main.py's /events endpoint calls
process_event() and nothing else.

LOGGING AT EVERY DROP POINT
-----------------------------
Every early return logs exactly which layer dropped the event and why.
This is essential for debugging "why didn't this alert show up?" --
a question operators and teammates will ask constantly. Without this,
a dropped event is indistinguishable from one that was never received.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from layers.dedup import run_dedup
from layers.enrichment import run_enrichment
from layers.suppression.exact_rules import run_exact_rule_check
from layers.suppression.semantic import run_semantic_suppression_check
from layers.threshold_gate import run_threshold_gate
from layers.tiering import build_dashboard_alert, run_tiering
from models.schemas import DashboardAlert, RawDetectionEvent

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Drop reason — for logging and (optionally) API responses
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PipelineDropReason:
    """
    Describes why an event was dropped, for structured logging.
    Not part of the public API contract (DashboardAlert / None) —
    purely an internal logging aid.
    """
    layer: str
    reason: str
    alert_id: str | None = None


# ─────────────────────────────────────────────────────────────
# The orchestrator
# ─────────────────────────────────────────────────────────────

async def process_event(
    raw_event: RawDetectionEvent,
    redis_client: Redis,
    db_session: AsyncSession,
    qdrant_client: AsyncQdrantClient,
) -> DashboardAlert | None:
    """
    Runs one RawDetectionEvent through all five layers in sequence.

    Parameters
    ----------
    raw_event:     The incoming detection from an upstream classifier.
    redis_client:  Async Redis client (Layer 3).
    db_session:    Async SQLAlchemy session (Layer 4A, alert_log write).
    qdrant_client: Async Qdrant client (Layer 4B).

    All three clients are injected rather than imported directly —
    this is what makes process_event() testable with fakeredis,
    a test Postgres database, and a test Qdrant collection, without
    touching production infrastructure.

    Returns
    -------
    DashboardAlert if the event survived all five layers, else None.
    """

    # ── Layer 1: Threshold Gate ──────────────────────────────
    gate_result = await run_threshold_gate(raw_event)

    if not gate_result.passed:
        logger.info(
            "pipeline.dropped",
            extra={
                "layer": "1_threshold_gate",
                "reason": "confidence_below_effective_threshold",
                "pipeline": raw_event.pipeline.value,
                "camera_id": raw_event.camera_id,
                "raw_confidence": raw_event.raw_confidence,
                "effective_threshold": gate_result.effective_threshold,
            },
        )
        return None

    # ── Layer 2: Enrichment ──────────────────────────────────
    # Pure data assembly — cannot fail/drop, only transforms.
    event = run_enrichment(raw_event, gate_result)

    # ── Layer 3: Burst Deduplication (Redis) ─────────────────
    dedup_result = await run_dedup(event, redis_client)

    if dedup_result.is_duplicate:
        logger.debug(
            "pipeline.dropped",
            extra={
                "layer": "3_dedup",
                "reason": "duplicate_of_active_incident",
                "alert_id": event.alert_id,
                "dedup_key": dedup_result.dedup_key,
            },
        )
        return None

    if dedup_result.is_escalation:
        logger.info(
            "pipeline.escalation",
            extra={
                "alert_id": event.alert_id,
                "dedup_key": dedup_result.dedup_key,
                "stored_confidence": dedup_result.stored_confidence,
            },
        )

    # ── Layer 4A: Exact Rule Suppression (Postgres) ──────────
    exact_result = await run_exact_rule_check(event, db_session)

    if exact_result.suppressed:
        logger.info(
            "pipeline.dropped",
            extra={
                "layer": "4a_exact_rules",
                "reason": "matched_explicit_suppression_rule",
                "alert_id": event.alert_id,
                "matched_rule_id": exact_result.matched_rule_id,
            },
        )
        return None

    # ── Layer 4B: Semantic Suppression (Qdrant) ──────────────
    semantic_result = await run_semantic_suppression_check(event, qdrant_client)

    if semantic_result.suppressed:
        logger.info(
            "pipeline.dropped",
            extra={
                "layer": "4b_semantic",
                "reason": "matched_dismissed_alert_pattern",
                "alert_id": event.alert_id,
                "nearest_score": semantic_result.nearest_score,
                "nearest_alert_id": semantic_result.nearest_alert_id,
            },
        )
        return None

    # ── Layer 5: Tier Assignment ──────────────────────────────
    tier_result = await run_tiering(event)
    dashboard_alert = build_dashboard_alert(
        event,
        tier_result.tier,
        suppression_score=semantic_result.nearest_score,
    )

    # ── Persist to alert_log (audit trail) ───────────────────
    await _write_alert_log(dashboard_alert, db_session)

    logger.info(
        "pipeline.alert_emitted",
        extra={
            "alert_id": dashboard_alert.alert_id,
            "pipeline": dashboard_alert.pipeline.value,
            "tier": dashboard_alert.tier.value,
            "camera_id": dashboard_alert.camera_id,
            "suppression_score": dashboard_alert.suppression_score,
        },
    )

    return dashboard_alert


# ─────────────────────────────────────────────────────────────
# alert_log writer
# ─────────────────────────────────────────────────────────────

async def _write_alert_log(
    alert: DashboardAlert,
    session: AsyncSession,
) -> None:
    """
    Writes a surviving alert to the alert_log hypertable.

    Uses ON CONFLICT DO NOTHING on alert_id — this is the idempotency
    guarantee from §4.10 in action. If the same underlying detection
    is delivered twice (message bus at-least-once delivery), the second
    INSERT silently no-ops rather than creating a duplicate row or
    raising an integrity error that would crash the pipeline.
    """
    stmt = text("""
        INSERT INTO alert_log
            (alert_id, pipeline, tier, camera_id, zone_id, zone_label,
             raw_confidence, effective_conf, timestamp, evidence_frame_ref,
             involved_ids, suppression_score)
        VALUES
            (:alert_id, :pipeline, :tier, :camera_id, :zone_id, :zone_label,
             :raw_confidence, :effective_conf, :timestamp, :evidence_frame_ref,
             :involved_ids, :suppression_score)
        ON CONFLICT (alert_id, timestamp) DO NOTHING
    """)

    await session.execute(
        stmt,
        {
            "alert_id":           alert.alert_id,
            "pipeline":           alert.pipeline.value,
            "tier":               alert.tier.value,
            "camera_id":          alert.camera_id,
            "zone_id":            alert.zone_id,
            "zone_label":         alert.zone_label,
            "raw_confidence":     alert.raw_confidence,
            "effective_conf":     alert.effective_conf,
            "timestamp":          alert.timestamp,
            "evidence_frame_ref": alert.evidence_frame_ref,
            "involved_ids":       alert.involved_ids,
            "suppression_score":  alert.suppression_score,
        },
    )
    await session.commit()