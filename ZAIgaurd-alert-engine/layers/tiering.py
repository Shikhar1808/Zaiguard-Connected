"""
ZaiGuard Alert Engine — Layer 5: Tier Assignment
=================================================
Assigns a severity tier (CRITICAL / HIGH / MEDIUM / LOW) to every
AlertEvent that survived Layers 1–4.

POSITION IN PIPELINE
--------------------
Layer 5 is the final gate before an event becomes a DashboardAlert.
Every event that reaches here WILL be shown to the operator — this
layer only decides how prominently and urgently.

WHY A SEPARATE LAYER
--------------------
Tier assignment could technically happen inside pipeline.py directly.
It's a separate layer for the same reason enrichment is separate from
thresholding: the concern is different. Layers 1–4 are about deciding
whether to show an alert. Layer 5 is about deciding how to show it.
Separating them means tier logic can be tested, tuned, and reasoned
about independently.

HOW TIER IS DETERMINED
-----------------------
The tier_config table (seeded in seed.sql) maps
(pipeline, min_confidence) → tier. Rows are ordered by min_confidence
descending, so the first matching row wins — the highest applicable
tier for this confidence level.

    pipeline     min_confidence   tier
    fire         0.60             CRITICAL
    violence     0.90             CRITICAL
    violence     0.72             HIGH
    dog_attack   0.80             HIGH
    dog_attack   0.68             MEDIUM
    trespassing  0.78             MEDIUM
    accident     0.80             HIGH
    accident     0.70             MEDIUM

If no rule matches (confidence below all configured minimums),
the fallback tier is LOW. This is a safety net — an event that
passed Layer 1 at the base threshold always gets displayed somewhere,
never silently dropped at the tiering stage.

All tier rules are stored in Postgres and cached by ThresholdConfigLoader.
Changes made from the dashboard take effect within config_cache_ttl_seconds
(default 60s) without any code change or restart.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from config.thresholds import ThresholdConfigLoader, config_loader
from models.schemas import AlertEvent, AlertTier, DashboardAlert

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TierResult:
    """
    Return value of run_tiering().

    tier:
        The assigned severity level for this event.

    pipeline:
        Echoed back for convenient logging in pipeline.py —
        avoids having to reach into the event again.
    """
    tier: AlertTier
    pipeline: str


# ─────────────────────────────────────────────────────────────
# Tier assignment
# ─────────────────────────────────────────────────────────────

async def run_tiering(
    event: AlertEvent,
    loader: ThresholdConfigLoader = config_loader,
) -> TierResult:
    """
    Layer 5: assigns a severity tier to a surviving AlertEvent.

    This is an async function even though it does no I/O — it calls
    ensure_fresh() which is async (and a near-instant no-op when the
    cache is fresh). Keeping it async means pipeline.py can await it
    uniformly with all other layers without special-casing.

    Parameters
    ----------
    event:  AlertEvent from Layer 2 (has survived Layers 3 and 4)
    loader: Config loader — injected for testability, defaults to singleton

    Returns
    -------
    TierResult with the assigned AlertTier
    """
    await loader.ensure_fresh()

    pipeline_str = event.pipeline.value
    tier_str = loader.get_tier(pipeline_str, event.effective_conf)

    try:
        tier = AlertTier(tier_str)
    except ValueError:
        # get_tier() already falls back to "LOW" — this branch is a
        # final safety net in case someone stores an invalid string
        # in the tier_config table.
        logger.warning(
            "tiering.invalid_tier_value",
            extra={
                "alert_id": event.alert_id,
                "pipeline": pipeline_str,
                "raw_tier": tier_str,
            },
        )
        tier = AlertTier.LOW

    logger.debug(
        "tiering.assigned",
        extra={
            "alert_id":      event.alert_id,
            "pipeline":      pipeline_str,
            "effective_conf": event.effective_conf,
            "tier":          tier.value,
        },
    )

    return TierResult(tier=tier, pipeline=pipeline_str)


# ─────────────────────────────────────────────────────────────
# DashboardAlert builder — assembles the final output object
# ─────────────────────────────────────────────────────────────

def build_dashboard_alert(
    event: AlertEvent,
    tier: AlertTier,
    suppression_score: float = 0.0,
) -> DashboardAlert:
    """
    Constructs the DashboardAlert that pipeline.py emits.

    Kept here (in the tiering module) because tier assignment and
    final object assembly are the last two things that happen before
    an event leaves the Alert Engine — they belong together.

    Parameters
    ----------
    event:             AlertEvent that survived all five layers
    tier:              Tier from run_tiering()
    suppression_score: Nearest cosine similarity from Layer 4B
                       (0.0 if no similar dismissed alerts found)
    """
    return DashboardAlert(
        alert_id=event.alert_id,
        tier=tier,
        pipeline=event.pipeline,
        camera_id=event.camera_id,
        zone_id=event.zone_id,
        zone_label=event.zone_label,
        raw_confidence=event.raw_confidence,
        effective_conf=event.effective_conf,
        timestamp=event.timestamp,
        evidence_frame_ref=event.frame_ref,
        involved_ids=event.involved_ids,
        suppression_score=suppression_score,
    )