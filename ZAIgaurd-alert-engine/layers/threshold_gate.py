"""
ZaiGuard Alert Engine — Layer 1: Threshold Gate
=================================================
The first gate in the pipeline. Every raw detection event from every
upstream classifier passes through here first.

WHAT IT DOES
-------------
Computes an effective threshold for the incoming event — based on the
event type, the time of day, and the zone's risk level — and decides
whether the event's confidence is high enough to continue through the
pipeline.

Events that don't pass are dropped here, silently, with a log line.
Events that pass are handed to Layer 2 (enrichment) with their
effective_conf attached.

WHY IT'S A PURE FUNCTION
--------------------------
Layer 1 has no side effects. It reads from the config cache (which is
already in memory — no I/O) and returns a result. This makes it:
  - Trivially testable: give it an event, check the result
  - Trivially debuggable: same inputs always produce same outputs
  - Safe to run concurrently: no shared mutable state

The config_loader.ensure_fresh() call at the start is the only
async operation — and in the common case (cache is fresh) it's just
a time comparison, not a DB query.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from config.thresholds import ThresholdConfigLoader, config_loader
from models.schemas import Pipeline, RawDetectionEvent

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ThresholdResult:
    """
    Return value of run_threshold_gate().

    Using a dataclass instead of a plain tuple means the caller
    can write `result.passed` and `result.effective_threshold`
    rather than `result[0]` and `result[1]` — self-documenting
    and impossible to accidentally swap.

    frozen=True makes it immutable — a gate result should never
    be modified after it's produced.
    """
    passed: bool
    effective_conf: float          # raw_confidence, unchanged — stored for clarity
    effective_threshold: float     # the computed threshold it was compared against
    raw_confidence: float          # original value from the model
    pipeline: Pipeline
    camera_id: str
    zone_id: str
    hour_of_day: int


# ─────────────────────────────────────────────────────────────
# The gate
# ─────────────────────────────────────────────────────────────

async def run_threshold_gate(
    event: RawDetectionEvent,
    loader: ThresholdConfigLoader = config_loader,
) -> ThresholdResult:
    """
    Gate 1: determine whether this event's confidence clears the
    effective threshold for its pipeline, time of day, and zone.

    Parameters
    ----------
    event:
        The raw detection event from the upstream classifier.
    loader:
        The config loader to use. Defaults to the module-level
        singleton. Passed as a parameter (dependency injection)
        so tests can inject a pre-populated loader without needing
        a live database.

    Returns
    -------
    ThresholdResult with passed=True if the event clears the gate,
    False otherwise. The caller (pipeline.py) checks .passed and
    either continues or drops the event.
    """
    # Refresh config if stale. No-op in the common case (just a
    # time comparison). Only hits the DB when the TTL has expired.
    await loader.ensure_fresh()

    hour = event.timestamp.hour

    effective_threshold = loader.get_effective_threshold(
        pipeline=event.pipeline.value,
        hour=hour,
        zone_id=event.zone_id,
    )

    passed = event.raw_confidence >= effective_threshold

    if passed:
        logger.debug(
            "threshold_gate.pass",
            extra={
                "pipeline": event.pipeline.value,
                "camera_id": event.camera_id,
                "zone_id": event.zone_id,
                "raw_confidence": event.raw_confidence,
                "effective_threshold": round(effective_threshold, 4),
                "hour": hour,
            },
        )
    else:
        logger.debug(
            "threshold_gate.drop",
            extra={
                "pipeline": event.pipeline.value,
                "camera_id": event.camera_id,
                "zone_id": event.zone_id,
                "raw_confidence": event.raw_confidence,
                "effective_threshold": round(effective_threshold, 4),
                "hour": hour,
                "shortfall": round(effective_threshold - event.raw_confidence, 4),
            },
        )

    return ThresholdResult(
        passed=passed,
        effective_conf=event.raw_confidence,
        effective_threshold=effective_threshold,
        raw_confidence=event.raw_confidence,
        pipeline=event.pipeline,
        camera_id=event.camera_id,
        zone_id=event.zone_id,
        hour_of_day=hour,
    )