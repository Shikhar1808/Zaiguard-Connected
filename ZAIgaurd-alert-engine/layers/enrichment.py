"""
ZaiGuard Alert Engine — Layer 2: Event Enrichment
===================================================
The second gate — though calling it a "gate" is a slight misnomer
since it never drops events. Every event that passes Layer 1 passes
through Layer 2. Its job is pure data assembly, not decision-making.

WHAT IT DOES
-------------
Takes a RawDetectionEvent (from the classifier) and a ThresholdResult
(from Layer 1) and constructs a complete AlertEvent object — the
internal representation used by every layer from here onward.

The key things it adds:
  - alert_id: deterministic unique ID for this event (idempotency)
  - effective_conf: from the threshold result
  - hour_of_day, day_of_week: pre-extracted from the timestamp so
    downstream layers never need to re-parse it

WHY SEPARATE FROM LAYER 1
---------------------------
Layer 1 is a decision (pass/fail). Layer 2 is data transformation
(build the object). Separating them keeps each one small, single-
purpose, and independently testable. Layer 1 tests never need to
know what AlertEvent looks like. Layer 2 tests never need to care
about thresholds.

WHY PRE-EXTRACT TIME FIELDS
-----------------------------
`timestamp.hour` and `timestamp.weekday()` are trivial to compute,
but every layer downstream that needs them would have to call
`event.source_event.timestamp.hour` — verbose, and a potential
source of bugs if different layers make different assumptions about
timezone handling. Extracting them once here, from a timestamp that
is guaranteed to be UTC (enforced by the Pydantic validator in
schemas.py), means downstream layers just read `event.hour_of_day`.
"""

from __future__ import annotations

import logging

from layers.threshold_gate import ThresholdResult
from models.schemas import AlertEvent, RawDetectionEvent, build_alert_id

logger = logging.getLogger(__name__)


def run_enrichment(
    event: RawDetectionEvent,
    threshold_result: ThresholdResult,
) -> AlertEvent:
    """
    Layer 2: assemble a complete AlertEvent from a raw detection
    event and its threshold gate result.

    This is a synchronous function — no I/O, no external services,
    just data transformation. Kept sync deliberately: async functions
    have a small overhead from the event loop machinery, and there is
    nothing async to await here.

    Parameters
    ----------
    event:
        The original RawDetectionEvent from the upstream classifier.
        Preserved in full inside AlertEvent.source_event so that
        pipeline_features and frame_ref are available to downstream
        layers without them needing to know about RawDetectionEvent.
    threshold_result:
        The ThresholdResult from Layer 1. Provides effective_conf and
        confirms the event already passed the threshold check.

    Returns
    -------
    AlertEvent ready to be handed to Layer 3 (burst deduplication).
    """
    alert_id = build_alert_id(
        camera_id=event.camera_id,
        timestamp=event.timestamp,
        pipeline=event.pipeline,
    )

    enriched = AlertEvent(
        alert_id=alert_id,
        source_event=event,
        effective_conf=threshold_result.effective_conf,
        hour_of_day=event.timestamp.hour,
        day_of_week=event.timestamp.weekday(),   # 0=Monday, 6=Sunday
    )

    logger.debug(
        "enrichment.complete",
        extra={
            "alert_id": alert_id,
            "pipeline": event.pipeline.value,
            "camera_id": event.camera_id,
            "effective_conf": threshold_result.effective_conf,
            "hour_of_day": enriched.hour_of_day,
            "day_of_week": enriched.day_of_week,
        },
    )

    return enriched