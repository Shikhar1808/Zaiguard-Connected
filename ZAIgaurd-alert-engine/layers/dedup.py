"""
ZaiGuard Alert Engine — Layer 3: Burst Deduplication
======================================================
Prevents a single ongoing incident from flooding the dashboard
with hundreds of near-identical alerts.

THE PROBLEM
-----------
A 30-second fight detected at 15fps generates ~450 raw detection
events. Every one of them passes Layer 1 (high confidence, same
camera, same zone). Without this layer, the operator sees 450
identical alerts. With it, they see one — or two if the incident
escalates significantly mid-way through.

THE MECHANISM
-------------
When an event arrives, a dedup key is constructed:
    "{camera_id}:{zone_id}:{pipeline}"

Redis is checked for this key:
  - Key EXISTS and confidence has NOT jumped significantly
    → DUPLICATE — drop this event, return DedupResult(is_duplicate=True)
  - Key EXISTS but confidence HAS jumped significantly (escalation)
    → ESCALATION — let through, update stored confidence in Redis
  - Key DOES NOT EXIST
    → NEW INCIDENT — write key with TTL, let through

The TTL (per pipeline, from config) is the dedup window. After it
expires, the key vanishes and the next event of that type from that
camera starts a fresh alert.

HYSTERESIS
----------
The escalation check applies the hysteresis principle (§4.8 of the
architecture doc). Rather than re-alerting every time confidence
inches upward, we only re-alert when confidence jumps by at least
`escalation_delta` (default 0.15). This prevents chattering —
the rapid on/off oscillation that would happen if confidence hovers
just above and below any re-alert boundary.

Redis is used (not Postgres) because this check runs on every single
incoming event, potentially thousands per second. Redis reads/writes
take ~100 nanoseconds from memory vs ~100 microseconds from disk.
The native TTL feature (automatic key expiry) is exactly the primitive
this layer needs — zero cleanup code required.

KEY SCHEMA
----------
Dedup key:      "{camera_id}:{zone_id}:{pipeline}"   → stored confidence (float str)
Both keys use the pipeline-specific TTL from threshold_config.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from redis.asyncio import Redis

from config.thresholds import ThresholdConfigLoader, config_loader
from models.schemas import AlertEvent

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DedupResult:
    """
    Return value of run_dedup().

    is_duplicate:   True  → caller should drop this event
                    False → caller should continue to Layer 4

    is_escalation:  True  → this event passed because confidence
                            jumped significantly on an active incident.
                            pipeline.py can use this to annotate the
                            alert as an escalation on the dashboard.

    stored_confidence:  The confidence value now stored in Redis for
                        this dedup key. On a new incident this equals
                        the current event's confidence. On an escalation
                        it equals the new (higher) confidence. On a
                        duplicate it equals whatever was already stored.

    dedup_key:  The Redis key used — included for logging/debugging.
    """
    is_duplicate: bool
    is_escalation: bool
    stored_confidence: float
    dedup_key: str


# ─────────────────────────────────────────────────────────────
# Key builder
# ─────────────────────────────────────────────────────────────

def build_dedup_key(event: AlertEvent) -> str:
    """
    Constructs the Redis key for a given event.

    Deliberately NOT including timestamp in the key — the whole
    point is that events from the same camera/zone/pipeline within
    the TTL window are considered the same ongoing incident regardless
    of when exactly they arrived.

    Colon separator chosen because none of camera_id, zone_id, or
    pipeline values should ever contain colons — pipeline values are
    enum strings like "violence", zone_ids are configured strings.
    If your camera naming convention ever uses colons, change the
    separator here to something like "|" or "::" consistently.
    """
    return f"{event.camera_id}:{event.zone_id}:{event.pipeline.value}"


# ─────────────────────────────────────────────────────────────
# The deduplication gate
# ─────────────────────────────────────────────────────────────

async def run_dedup(
    event: AlertEvent,
    redis_client: Redis,
    loader: ThresholdConfigLoader = config_loader,
) -> DedupResult:
    """
    Layer 3: burst deduplication via Redis TTL keys.

    Parameters
    ----------
    event:
        The enriched AlertEvent from Layer 2.
    redis_client:
        Async Redis client. Injected rather than imported directly
        so tests can substitute fakeredis without touching real Redis.
    loader:
        Config loader for TTL and escalation_delta values. Injected
        for the same testability reason as redis_client.

    Returns
    -------
    DedupResult with is_duplicate=False if the event should continue
    to Layer 4, True if it should be dropped.
    """
    await loader.ensure_fresh()

    dedup_key = build_dedup_key(event)
    pipeline_str = event.pipeline.value
    current_conf = event.effective_conf

    ttl_seconds = loader.get_dedup_ttl(pipeline_str)
    escalation_delta = loader.get_escalation_delta(pipeline_str)

    # Check if this key already exists in Redis
    stored_value = await redis_client.get(dedup_key)

    if stored_value is None:
        # ── NEW INCIDENT ──────────────────────────────────────
        # No active dedup key for this camera/zone/pipeline.
        # Write the key with the pipeline-specific TTL and let
        # the event through.
        await redis_client.setex(
            name=dedup_key,
            time=ttl_seconds,
            value=str(current_conf),
        )

        logger.debug(
            "dedup.new_incident",
            extra={
                "dedup_key": dedup_key,
                "confidence": current_conf,
                "ttl_seconds": ttl_seconds,
            },
        )

        return DedupResult(
            is_duplicate=False,
            is_escalation=False,
            stored_confidence=current_conf,
            dedup_key=dedup_key,
        )

    # Key exists — active incident window
    stored_conf = float(stored_value)
    confidence_jump = current_conf - stored_conf

    if confidence_jump >= escalation_delta:
        # ── ESCALATION ────────────────────────────────────────
        # Confidence has jumped significantly. This indicates the
        # incident is getting worse, not just the same event being
        # re-detected. Let through and update the stored confidence.
        #
        # We use SETEX again (not just SET) to also reset the TTL —
        # an escalating incident should get a fresh window, not
        # expire imminently because the original TTL is almost up.
        await redis_client.setex(
            name=dedup_key,
            time=ttl_seconds,
            value=str(current_conf),
        )

        logger.info(
            "dedup.escalation",
            extra={
                "dedup_key": dedup_key,
                "previous_confidence": stored_conf,
                "new_confidence": current_conf,
                "jump": round(confidence_jump, 4),
                "escalation_delta": escalation_delta,
            },
        )

        return DedupResult(
            is_duplicate=False,
            is_escalation=True,
            stored_confidence=current_conf,
            dedup_key=dedup_key,
        )

    # ── DUPLICATE ─────────────────────────────────────────────
    # Active incident, no significant confidence jump.
    # Drop this event silently.
    logger.debug(
        "dedup.duplicate",
        extra={
            "dedup_key": dedup_key,
            "stored_confidence": stored_conf,
            "current_confidence": current_conf,
            "jump": round(confidence_jump, 4),
            "escalation_delta": escalation_delta,
        },
    )

    return DedupResult(
        is_duplicate=True,
        is_escalation=False,
        stored_confidence=stored_conf,
        dedup_key=dedup_key,
    )


# ─────────────────────────────────────────────────────────────
# Utility: clear a dedup key manually
# ─────────────────────────────────────────────────────────────

async def clear_dedup_key(
    event: AlertEvent,
    redis_client: Redis,
) -> bool:
    """
    Manually clears the dedup key for an event, allowing the next
    detection of that type from that camera to pass through as a
    new incident regardless of TTL.

    Called by the feedback handler (Step 10) when an operator
    confirms an alert and wants to keep receiving updates on an
    ongoing incident — clearing the key means the next detection
    fires a fresh alert rather than being suppressed as a duplicate.

    Returns True if a key was deleted, False if it didn't exist.
    """
    dedup_key = build_dedup_key(event)
    deleted = await redis_client.delete(dedup_key)

    logger.debug(
        "dedup.key_cleared",
        extra={
            "dedup_key": dedup_key,
            "was_present": bool(deleted),
        },
    )

    return bool(deleted)