"""
ZaiGuard Alert Engine — Feedback Handler
=========================================
Processes operator feedback (dismiss / confirm) from the dashboard.

WHAT THIS FILE OWNS
--------------------
When an operator acts on a DashboardAlert, two things must happen
depending on the action:

DISMISS:
  1. Write a suppression_rules row to Postgres         (Layer 4A future)
  2. Write an embedding to Qdrant dismissed_alerts     (Layer 4B future)
  3. Write the Qdrant write as an outbox row           (consistency)
  Optionally:
  4. Clear the Redis dedup key if re-alerting wanted

CONFIRM:
  1. Update alert_log.operator_action = 'confirmed'
  2. Optionally clear the Redis dedup key so the incident keeps
     producing alerts without the burst dedup window suppressing them

THE OUTBOX PATTERN (§4.1 of the architecture document)
-------------------------------------------------------
We cannot atomically write to two independent databases (Postgres +
Qdrant). The outbox pattern solves this:

  Step A: In ONE Postgres transaction, write BOTH:
    - The suppression_rules row
    - An outbox row describing the pending Qdrant write

  Step B: A background worker (process_outbox_batch) reads unprocessed
  outbox rows and performs the actual Qdrant upsert, marking each row
  as processed on success and incrementing failed_attempts on failure.

If Step A fails → nothing happened, retrying is safe.
If Step A succeeds but Step B hasn't run yet → the rule exists in
  Postgres (Layer 4A will use it) but the embedding isn't in Qdrant
  yet (Layer 4B won't use it for ~seconds). That's acceptable —
  "eventually consistent" is fine for semantic suppression.
If Step B fails on Qdrant → failed_attempts increments, the row stays
  in the outbox, and the worker retries on the next batch. The outbox
  row is the source of truth until confirmed processed.

This means the handler never calls Qdrant directly — only the outbox
worker does. This is the correct separation.

TTL POLICY (mirrors Layer 4A)
------------------------------
  permanent=False: suppression_rules expires in 24h, Qdrant TTL 24h
  permanent=True:  suppression_rules expires never, Qdrant is_permanent=True
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from layers.dedup import build_dedup_key
from layers.suppression.exact_rules import write_suppression_rule
from models.schemas import AlertEvent, DashboardAlert, OperatorAction, OperatorFeedback, RawDetectionEvent

logger = logging.getLogger(__name__)

# Default TTL for non-permanent dismissals — must match semantic.py
DISMISSAL_TTL_HOURS = 24


# ─────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────

async def process_feedback(
    feedback: OperatorFeedback,
    alert: DashboardAlert,
    session: AsyncSession,
    redis_client: Redis | None = None,
) -> dict:
    """
    Processes operator feedback for a DashboardAlert.

    Parameters
    ----------
    feedback:     The OperatorFeedback from the dashboard endpoint.
    alert:        The DashboardAlert being acted on. The dashboard
                  must send the full alert object (or enough fields)
                  for us to build the suppression rule and embedding.
                  In practice main.py retrieves the alert from
                  alert_log before calling this.
    session:      Async SQLAlchemy session. The Postgres writes
                  (suppression_rules + outbox) happen in ONE transaction
                  via this session — that's the outbox pattern guarantee.
    redis_client: Optional. If provided, the dedup key for this alert
                  is cleared on CONFIRM so the incident keeps producing
                  fresh alerts without the burst window blocking them.

    Returns
    -------
    A dict summarising what was done, returned to the caller.
    """
    if feedback.action == OperatorAction.DISMISS:
        return await _handle_dismiss(feedback, alert, session, redis_client)
    else:
        return await _handle_confirm(feedback, alert, session, redis_client)


# ─────────────────────────────────────────────────────────────
# Dismiss path
# ─────────────────────────────────────────────────────────────

async def _handle_dismiss(
    feedback: OperatorFeedback,
    alert: DashboardAlert,
    session: AsyncSession,
    redis_client: Redis | None,
) -> dict:
    """
    Dismiss: write suppression rule + outbox row in ONE transaction.

    The outbox row payload contains everything the worker needs to
    call store_dismissed_alert() later — it does not need to re-query
    Postgres for the alert's details.
    """
    permanent = feedback.permanent
    expires_at = (
        None if permanent
        else datetime.now(timezone.utc) + timedelta(hours=DISMISSAL_TTL_HOURS)
    )

    # Build outbox payload before the transaction so we don't
    # call any I/O inside the transaction block itself.
    outbox_payload = _build_outbox_payload(alert, permanent)

    # ── Single Postgres transaction: rule + outbox ────────────
    # Both inserts happen atomically. If either fails, neither
    # commits — no partial state, safe to retry the whole dismiss.
    rule_id = await _write_dismiss_transaction(
        alert=alert,
        expires_at=expires_at,
        permanent=permanent,
        outbox_payload=outbox_payload,
        session=session,
    )

    logger.info(
        "feedback.dismissed",
        extra={
            "alert_id": feedback.alert_id,
            "rule_id": rule_id,
            "permanent": permanent,
            "expires_at": expires_at.isoformat() if expires_at else "permanent",
        },
    )

    return {
        "status": "dismissed",
        "alert_id": feedback.alert_id,
        "suppression_rule_id": rule_id,
        "permanent": permanent,
        "expires_at": expires_at.isoformat() if expires_at else None,
    }


async def _write_dismiss_transaction(
    alert: DashboardAlert,
    expires_at: datetime | None,
    permanent: bool,
    outbox_payload: dict,
    session: AsyncSession,
) -> str:
    """
    Writes suppression_rules row + outbox row in a single transaction.

    Returns the newly created rule UUID.
    """
    # 1. Write the suppression rule
    rule_stmt = text("""
        INSERT INTO suppression_rules
            (camera_id, pipeline, zone_id, hour_start, hour_end,
             days_mask, expires_at, source, created_at)
        VALUES
            (:camera_id, :pipeline, :zone_id, NULL, NULL,
             NULL, :expires_at, 'manual', :created_at)
        RETURNING id::text
    """)

    rule_result = await session.execute(
        rule_stmt,
        {
            "camera_id":  alert.camera_id,
            "pipeline":   alert.pipeline.value,
            "zone_id":    alert.zone_id,
            "expires_at": expires_at,
            "created_at": datetime.now(timezone.utc),
        },
    )
    rule_id = rule_result.fetchone()[0]

    # 2. Write the outbox row (pending Qdrant write)
    outbox_stmt = text("""
        INSERT INTO outbox (event_type, payload, created_at)
        VALUES ('dismissed_alert_embedding', :payload, :created_at)
    """)

    await session.execute(
        outbox_stmt,
        {
            "payload":    json.dumps(outbox_payload),
            "created_at": datetime.now(timezone.utc),
        },
    )

    # Both inserts commit together — this is the atomic guarantee
    await session.commit()
    return rule_id


def _build_outbox_payload(alert: DashboardAlert, permanent: bool) -> dict:
    """
    Builds the JSON payload stored in the outbox row.

    Contains everything the outbox worker needs to call
    store_dismissed_alert() without any additional DB lookups.
    """
    ttl_expires = (
        None if permanent
        else (
            datetime.now(timezone.utc) + timedelta(hours=DISMISSAL_TTL_HOURS)
        ).isoformat()
    )
    return {
        "alert_id":    alert.alert_id,
        "pipeline":    alert.pipeline.value,
        "camera_id":   alert.camera_id,
        "zone_id":     alert.zone_id,
        "zone_label":  alert.zone_label,
        "timestamp":   alert.timestamp.isoformat(),
        "is_permanent": permanent,
        "ttl_expires":  ttl_expires,
    }


# ─────────────────────────────────────────────────────────────
# Confirm path
# ─────────────────────────────────────────────────────────────

async def _handle_confirm(
    feedback: OperatorFeedback,
    alert: DashboardAlert,
    session: AsyncSession,
    redis_client: Redis | None,
) -> dict:
    """
    Confirm: update alert_log.operator_action, optionally clear dedup key.
    """
    # Mark alert as confirmed in alert_log
    await session.execute(
        text("""
            UPDATE alert_log
            SET operator_action = 'confirmed',
                action_at = :action_at
            WHERE alert_id = :alert_id
        """),
        {
            "alert_id":  feedback.alert_id,
            "action_at": datetime.now(timezone.utc),
        },
    )
    await session.commit()

    # Clear the Redis dedup key so the ongoing incident keeps
    # producing fresh alerts rather than being silenced by burst dedup
    dedup_key_cleared = False
    if redis_client is not None:
        dedup_key = f"{alert.camera_id}:{alert.zone_id}:{alert.pipeline.value}"
        deleted = await redis_client.delete(dedup_key)
        dedup_key_cleared = bool(deleted)

    logger.info(
        "feedback.confirmed",
        extra={
            "alert_id":         feedback.alert_id,
            "dedup_key_cleared": dedup_key_cleared,
        },
    )

    return {
        "status": "confirmed",
        "alert_id": feedback.alert_id,
        "dedup_key_cleared": dedup_key_cleared,
    }


# ─────────────────────────────────────────────────────────────
# Outbox worker — processes pending Qdrant writes
# ─────────────────────────────────────────────────────────────

MAX_FAILED_ATTEMPTS = 5  # give up after this many Qdrant failures


async def process_outbox_batch(
    session: AsyncSession,
    qdrant_client,
    batch_size: int = 50,
) -> int:
    """
    Reads unprocessed outbox rows and performs the pending Qdrant writes.

    Called periodically by a background task (added to main.py in this
    step) — not on every request. A batch_size of 50 means at most 50
    Qdrant upserts per invocation.

    Returns the number of rows successfully processed.
    """
    from layers.suppression.semantic import store_dismissed_alert, SemanticEmbedder

    # Fetch a batch of unprocessed rows, oldest first
    rows_result = await session.execute(
        text("""
            SELECT id, event_type, payload
            FROM outbox
            WHERE processed_at IS NULL
              AND failed_attempts < :max_attempts
            ORDER BY created_at ASC
            LIMIT :batch_size
        """),
        {
            "max_attempts": MAX_FAILED_ATTEMPTS,
            "batch_size":   batch_size,
        },
    )
    rows = rows_result.fetchall()

    if not rows:
        return 0

    processed = 0

    for row_id, event_type, payload_str in rows:
        if event_type != "dismissed_alert_embedding":
            # Unknown event type — mark processed to avoid re-processing
            await _mark_outbox_processed(session, row_id)
            continue

        try:
            payload = payload_str if isinstance(payload_str, dict) else json.loads(payload_str)
            await _process_embedding_row(payload, qdrant_client)
            await _mark_outbox_processed(session, row_id)
            processed += 1

        except Exception as e:
            await _mark_outbox_failed(session, row_id, str(e))
            logger.error(
                "outbox.worker.failed",
                extra={"outbox_id": row_id, "error": str(e)},
                exc_info=True,
            )

    await session.commit()

    logger.info(
        "outbox.worker.batch_complete",
        extra={"processed": processed, "total_rows": len(rows)},
    )

    return processed


async def _process_embedding_row(payload: dict, qdrant_client) -> None:
    """
    Reconstructs the minimum event data needed for store_dismissed_alert()
    from the outbox payload and performs the Qdrant upsert.
    """
    from layers.suppression.semantic import store_dismissed_alert, SemanticEmbedder
    from models.schemas import Pipeline

    # Reconstruct a minimal RawDetectionEvent-like object from payload.
    # We only need the fields that build_alert_description() reads —
    # pipeline, camera_id, zone_id, zone_label, timestamp, and
    # pipeline_features (empty dict is fine for dismissed alerts since
    # the description was already built when the alert fired).
    pipeline = Pipeline(payload["pipeline"])

    raw = RawDetectionEvent(
        pipeline=pipeline,
        raw_confidence=0.0,          # not needed for embedding
        camera_id=payload["camera_id"],
        zone_id=payload["zone_id"],
        zone_label=payload["zone_label"],
        timestamp=datetime.fromisoformat(payload["timestamp"]),
        pipeline_features={},
    )

    await store_dismissed_alert(
        client=qdrant_client,
        alert_id=payload["alert_id"],
        event=raw,
        permanent=payload.get("is_permanent", False),
    )


async def _mark_outbox_processed(session: AsyncSession, row_id: int) -> None:
    await session.execute(
        text("""
            UPDATE outbox
            SET processed_at = :now
            WHERE id = :id
        """),
        {"now": datetime.now(timezone.utc), "id": row_id},
    )


async def _mark_outbox_failed(
    session: AsyncSession, row_id: int, error: str
) -> None:
    await session.execute(
        text("""
            UPDATE outbox
            SET failed_attempts = failed_attempts + 1,
                last_error = :error
            WHERE id = :id
        """),
        {"error": error[:500], "id": row_id},
    )