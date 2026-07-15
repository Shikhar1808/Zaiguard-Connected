"""
ZaiGuard Alert Engine — Layer 4A: Exact Rule Suppression
=========================================================
Checks whether an operator has explicitly configured a suppression
rule that matches this event's camera, pipeline, time, and day.

POSITION IN PIPELINE
---------------------
Layer 4A runs AFTER burst dedup (Layer 3) and BEFORE semantic
suppression (Layer 4B). It runs first among the suppression layers
because a SQL index lookup (~1ms) is cheaper than a vector ANN
search (~5ms). Cheap, high-confidence checks always run before
expensive, probabilistic ones.

HOW RULES ARE CREATED
----------------------
Two sources:
1. Manual — operator explicitly creates a rule from the dashboard
   (e.g. "suppress violence from CAM_07 in gym_east 16:00–19:00
   on weekdays"). These have source='manual'.
2. Auto-promoted — the background job in scripts/auto_promote.py
   detects when the same pattern has been dismissed 5+ times and
   automatically promotes it to a permanent rule. source='auto_promoted'.

Both are stored identically in suppression_rules and checked the
same way here.

THE BITMASK CHECK
-----------------
days_mask stores which days a rule applies to as a bitmask:
    Mon=1, Tue=2, Wed=4, Thu=8, Fri=16, Sat=32, Sun=64
    Weekdays = 1+2+4+8+16 = 31

To check if a given day_of_week (0=Mon...6=Sun) is in the mask:
    days_mask & (1 << day_of_week) > 0

This is a single arithmetic operation inside Postgres — faster than
a JOIN against a separate (rule_id, day) table and semantically
cleaner to read.

THE EXPIRY CHECK
----------------
Rules have an optional expires_at timestamp:
  - NULL      → permanent rule, never expires
  - timestamp → rule expires at that time (24h default for dismissals,
                configurable from the dashboard)

The query filters out expired rules so we never suppress on a rule
the operator intended to be temporary.

QUERY PERFORMANCE
-----------------
The composite index idx_suppression_camera_pipeline on
(camera_id, pipeline) was created in schemas.sql. Postgres uses this
index to jump directly to the relevant rules for this camera+pipeline
combination, then applies the hour/day/expiry filters on that small
subset. Even with millions of rules accumulated over years, this
lookup stays sub-millisecond.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import and_, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from models.schemas import AlertEvent

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ExactRuleResult:
    """
    Return value of run_exact_rule_check().

    suppressed:      True  → drop this event, a rule matched
                     False → continue to Layer 4B

    matched_rule_id: UUID string of the rule that triggered
                     suppression, or None if no rule matched.
                     Included so pipeline.py can log which rule
                     caused suppression — dashboard can show
                     "suppressed by rule X" rather than silent drop.
    """
    suppressed: bool
    matched_rule_id: str | None = None


# ─────────────────────────────────────────────────────────────
# The suppression check
# ─────────────────────────────────────────────────────────────

async def run_exact_rule_check(
    event: AlertEvent,
    session: AsyncSession,
) -> ExactRuleResult:
    """
    Layer 4A: checks suppression_rules for any active rule
    matching this event's camera, pipeline, time, and day.

    Parameters
    ----------
    event:
        The enriched AlertEvent from Layer 2.
    session:
        Async SQLAlchemy session. Injected so this function is
        testable without constructing a full app context — tests
        pass a session pointing at a test database directly.

    Returns
    -------
    ExactRuleResult with suppressed=True if any rule matched.
    """
    now = datetime.now(timezone.utc)
    pipeline_str = event.pipeline.value
    hour = event.hour_of_day
    # day_of_week: 0=Monday...6=Sunday (Python convention)
    # Bitmask: 1 << 0 = 1 (Mon), 1 << 6 = 64 (Sun)
    day_bit = 1 << event.day_of_week

    # Raw SQL for clarity and auditability.
    # The ORM equivalent would work but this reads exactly like the
    # schema comment in schemas.sql — easier to review during integration.
    #
    # Logic per column:
    #   camera_id   — must match exactly
    #   pipeline    — must match exactly
    #   hour window — if hour_start/end are set, current hour must be
    #                 within range; if NULL, rule applies at all hours
    #   days_mask   — if set, current day must be in the bitmask;
    #                 if NULL, rule applies on all days
    #   expires_at  — if set, must be in the future; NULL = permanent
    stmt = text("""
        SELECT id::text
        FROM suppression_rules
        WHERE camera_id   = :camera_id
          AND pipeline    = :pipeline
          AND (
                hour_start IS NULL
                OR (hour_start <= :hour AND hour_end >= :hour)
              )
          AND (
                days_mask IS NULL
                OR (days_mask & :day_bit) > 0
              )
          AND (
                expires_at IS NULL
                OR expires_at > :now
              )
        LIMIT 1
    """)

    result = await session.execute(
        stmt,
        {
            "camera_id": event.camera_id,
            "pipeline":  pipeline_str,
            "hour":      hour,
            "day_bit":   day_bit,
            "now":       now,
        },
    )

    row = result.fetchone()

    if row is not None:
        matched_id = row[0]
        logger.info(
            "suppression.exact_rule.matched",
            extra={
                "alert_id":       event.alert_id,
                "camera_id":      event.camera_id,
                "pipeline":       pipeline_str,
                "rule_id":        matched_id,
                "hour":           hour,
                "day_of_week":    event.day_of_week,
            },
        )
        return ExactRuleResult(suppressed=True, matched_rule_id=matched_id)

    logger.debug(
        "suppression.exact_rule.no_match",
        extra={
            "alert_id":  event.alert_id,
            "camera_id": event.camera_id,
            "pipeline":  pipeline_str,
        },
    )
    return ExactRuleResult(suppressed=False, matched_rule_id=None)


# ─────────────────────────────────────────────────────────────
# Rule writer — used by feedback/handler.py (Step 10)
# ─────────────────────────────────────────────────────────────

async def write_suppression_rule(
    camera_id: str,
    pipeline: str,
    zone_id: str | None,
    hour_start: int | None,
    hour_end: int | None,
    days_mask: int | None,
    expires_at: datetime | None,
    source: str,
    session: AsyncSession,
) -> str:
    """
    Inserts a new suppression rule and returns its UUID.

    Called by feedback/handler.py (Step 10) when:
    - An operator dismisses an alert (source='manual')
    - auto_promote.py promotes a repeated pattern (source='auto_promoted')

    Returns the UUID of the newly created rule so the feedback
    handler can include it in the outbox payload for Qdrant write-back.
    """
    stmt = text("""
        INSERT INTO suppression_rules
            (camera_id, pipeline, zone_id, hour_start, hour_end,
             days_mask, expires_at, source, created_at)
        VALUES
            (:camera_id, :pipeline, :zone_id, :hour_start, :hour_end,
             :days_mask, :expires_at, :source, :created_at)
        RETURNING id::text
    """)

    result = await session.execute(
        stmt,
        {
            "camera_id":  camera_id,
            "pipeline":   pipeline,
            "zone_id":    zone_id,
            "hour_start": hour_start,
            "hour_end":   hour_end,
            "days_mask":  days_mask,
            "expires_at": expires_at,
            "source":     source,
            "created_at": datetime.now(timezone.utc),
        },
    )
    await session.commit()

    rule_id = result.fetchone()[0]

    logger.info(
        "suppression.exact_rule.written",
        extra={
            "rule_id":    rule_id,
            "camera_id":  camera_id,
            "pipeline":   pipeline,
            "source":     source,
            "expires_at": expires_at.isoformat() if expires_at else "permanent",
        },
    )

    return rule_id