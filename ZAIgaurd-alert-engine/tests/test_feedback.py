"""
Tests for feedback/handler.py (Step 10)
=========================================
Tests the full feedback loop — dismiss writes suppression rule +
outbox row, confirm updates alert_log, outbox worker processes
pending Qdrant writes.

Requires Docker running with all three services healthy:
    docker compose ps
    pytest tests/test_feedback.py -v
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels
from redis.asyncio import Redis
from sqlalchemy import text

from config.database import db_session_scope
from config.settings import settings
from config.thresholds import config_loader
from feedback.handler import (
    DISMISSAL_TTL_HOURS,
    process_feedback,
    process_outbox_batch,
)
from layers.suppression.semantic import (
    _to_point_id,
    ensure_qdrant_collections,
)
from models.schemas import (
    AlertTier,
    DashboardAlert,
    OperatorAction,
    OperatorFeedback,
    Pipeline,
)


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest_asyncio.fixture(scope="session", autouse=True)
async def setup_config():
    await config_loader.ensure_fresh()


@pytest_asyncio.fixture(scope="session")
async def redis_client():
    client = Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        decode_responses=True,
    )
    yield client
    await client.aclose()


@pytest_asyncio.fixture(scope="session")
async def qdrant_client():
    client = AsyncQdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    await ensure_qdrant_collections(client)
    yield client
    await client.close()


def make_alert(
    pipeline: Pipeline = Pipeline.VIOLENCE,
    camera_id: str = "CAM_FB_TEST_01",
    zone_id: str = "gym_east",
    confidence: float = 0.88,
) -> DashboardAlert:
    alert_id = uuid.uuid4().hex[:32]
    return DashboardAlert(
        alert_id=alert_id,
        tier=AlertTier.HIGH,
        pipeline=pipeline,
        camera_id=camera_id,
        zone_id=zone_id,
        zone_label="Gym East",
        raw_confidence=confidence,
        effective_conf=confidence,
        timestamp=datetime(2026, 6, 8, 17, 0, 0, tzinfo=timezone.utc),
        evidence_frame_ref="/clips/test.mp4",
        involved_ids=[1, 2],
        suppression_score=0.0,
    )


async def cleanup_rule_by_camera(camera_id: str) -> None:
    async with db_session_scope() as session:
        await session.execute(
            text("DELETE FROM suppression_rules WHERE camera_id = :c"),
            {"c": camera_id},
        )


async def cleanup_outbox_by_camera(camera_id: str) -> None:
    async with db_session_scope() as session:
        await session.execute(
            text("""
                DELETE FROM outbox
                WHERE payload::jsonb->>'camera_id' = :c
            """),
            {"c": camera_id},
        )


async def cleanup_alert_log(alert_id: str) -> None:
    async with db_session_scope() as session:
        await session.execute(
            text("DELETE FROM alert_log WHERE alert_id = :id"),
            {"id": alert_id},
        )


async def insert_alert_log(alert: DashboardAlert) -> None:
    async with db_session_scope() as session:
        await session.execute(
            text("""
                INSERT INTO alert_log
                    (alert_id, pipeline, tier, camera_id, zone_id, zone_label,
                     raw_confidence, effective_conf, timestamp)
                VALUES
                    (:alert_id, :pipeline, :tier, :camera_id, :zone_id, :zone_label,
                     :raw_confidence, :effective_conf, :timestamp)
                ON CONFLICT (alert_id, timestamp) DO NOTHING
            """),
            {
                "alert_id":       alert.alert_id,
                "pipeline":       alert.pipeline.value,
                "tier":           alert.tier.value,
                "camera_id":      alert.camera_id,
                "zone_id":        alert.zone_id,
                "zone_label":     alert.zone_label,
                "raw_confidence": alert.raw_confidence,
                "effective_conf": alert.effective_conf,
                "timestamp":      alert.timestamp,
            },
        )


# ─────────────────────────────────────────────────────────────
# Dismiss — suppression rule
# ─────────────────────────────────────────────────────────────

class TestDismissCreatesSuppressionRule:

    async def test_dismiss_creates_rule_in_postgres(self):
        alert = make_alert(camera_id="CAM_FB_RULE_01")
        feedback = OperatorFeedback(
            alert_id=alert.alert_id, action=OperatorAction.DISMISS
        )

        try:
            async with db_session_scope() as session:
                result = await process_feedback(feedback, alert, session)

            assert result["status"] == "dismissed"
            assert result["suppression_rule_id"] is not None

            # Verify rule exists in Postgres
            async with db_session_scope() as session:
                row = await session.execute(
                    text("""
                        SELECT camera_id, pipeline, expires_at
                        FROM suppression_rules
                        WHERE camera_id = :c AND pipeline = :p
                    """),
                    {"c": alert.camera_id, "p": alert.pipeline.value},
                )
                rule = row.fetchone()

            assert rule is not None
            assert rule[0] == alert.camera_id
            assert rule[1] == "violence"
            # Non-permanent: expires_at should be set
            assert rule[2] is not None

        finally:
            await cleanup_rule_by_camera("CAM_FB_RULE_01")
            await cleanup_outbox_by_camera("CAM_FB_RULE_01")

    async def test_dismiss_permanent_creates_rule_with_no_expiry(self):
        alert = make_alert(camera_id="CAM_FB_PERM_01")
        feedback = OperatorFeedback(
            alert_id=alert.alert_id,
            action=OperatorAction.DISMISS,
            permanent=True,
        )

        try:
            async with db_session_scope() as session:
                result = await process_feedback(feedback, alert, session)

            async with db_session_scope() as session:
                row = await session.execute(
                    text("""
                        SELECT expires_at FROM suppression_rules
                        WHERE camera_id = :c
                    """),
                    {"c": alert.camera_id},
                )
                rule = row.fetchone()

            assert rule is not None
            assert rule[0] is None   # NULL = permanent

        finally:
            await cleanup_rule_by_camera("CAM_FB_PERM_01")
            await cleanup_outbox_by_camera("CAM_FB_PERM_01")

    async def test_dismiss_ttl_is_approximately_24h(self):
        alert = make_alert(camera_id="CAM_FB_TTL_01")
        feedback = OperatorFeedback(
            alert_id=alert.alert_id, action=OperatorAction.DISMISS
        )

        try:
            async with db_session_scope() as session:
                await process_feedback(feedback, alert, session)

            async with db_session_scope() as session:
                row = await session.execute(
                    text("""
                        SELECT expires_at FROM suppression_rules
                        WHERE camera_id = :c
                    """),
                    {"c": alert.camera_id},
                )
                rule = row.fetchone()

            expires_at = rule[0]
            diff = expires_at - datetime.now(timezone.utc)
            assert timedelta(hours=23) < diff < timedelta(hours=25)

        finally:
            await cleanup_rule_by_camera("CAM_FB_TTL_01")
            await cleanup_outbox_by_camera("CAM_FB_TTL_01")


# ─────────────────────────────────────────────────────────────
# Dismiss — outbox row (Outbox Pattern)
# ─────────────────────────────────────────────────────────────

class TestDismissCreatesOutboxRow:

    async def test_dismiss_writes_outbox_row(self):
        alert = make_alert(camera_id="CAM_FB_OUTBOX_01")
        feedback = OperatorFeedback(
            alert_id=alert.alert_id, action=OperatorAction.DISMISS
        )

        try:
            async with db_session_scope() as session:
                await process_feedback(feedback, alert, session)

            async with db_session_scope() as session:
                row = await session.execute(
                    text("""
                        SELECT event_type, payload, processed_at
                        FROM outbox
                        WHERE payload::jsonb->>'camera_id' = :c
                        ORDER BY created_at DESC
                        LIMIT 1
                    """),
                    {"c": alert.camera_id},
                )
                outbox_row = row.fetchone()

            assert outbox_row is not None
            assert outbox_row[0] == "dismissed_alert_embedding"
            assert outbox_row[2] is None  # not yet processed

            payload = outbox_row[1]
            if isinstance(payload, str):
                payload = json.loads(payload)
            assert payload["alert_id"] == alert.alert_id
            assert payload["pipeline"] == "violence"
            assert payload["camera_id"] == alert.camera_id

        finally:
            await cleanup_rule_by_camera("CAM_FB_OUTBOX_01")
            await cleanup_outbox_by_camera("CAM_FB_OUTBOX_01")

    async def test_outbox_payload_has_correct_ttl_fields(self):
        alert = make_alert(camera_id="CAM_FB_OUTBOX_02")
        feedback = OperatorFeedback(
            alert_id=alert.alert_id,
            action=OperatorAction.DISMISS,
            permanent=True,
        )

        try:
            async with db_session_scope() as session:
                await process_feedback(feedback, alert, session)

            async with db_session_scope() as session:
                row = await session.execute(
                    text("""
                        SELECT payload FROM outbox
                        WHERE payload::jsonb->>'camera_id' = :c
                        ORDER BY created_at DESC LIMIT 1
                    """),
                    {"c": alert.camera_id},
                )
                outbox_row = row.fetchone()

            payload = outbox_row[0]
            if isinstance(payload, str):
                payload = json.loads(payload)

            assert payload["is_permanent"] is True
            assert payload["ttl_expires"] is None

        finally:
            await cleanup_rule_by_camera("CAM_FB_OUTBOX_02")
            await cleanup_outbox_by_camera("CAM_FB_OUTBOX_02")


# ─────────────────────────────────────────────────────────────
# Confirm path
# ─────────────────────────────────────────────────────────────

class TestConfirmUpdatesAlertLog:

    async def test_confirm_updates_operator_action_in_alert_log(
        self, redis_client
    ):
        alert = make_alert(camera_id="CAM_FB_CONFIRM_01")
        await insert_alert_log(alert)

        feedback = OperatorFeedback(
            alert_id=alert.alert_id, action=OperatorAction.CONFIRM
        )

        try:
            async with db_session_scope() as session:
                result = await process_feedback(
                    feedback, alert, session, redis_client
                )

            assert result["status"] == "confirmed"

            async with db_session_scope() as session:
                row = await session.execute(
                    text("""
                        SELECT operator_action, action_at
                        FROM alert_log WHERE alert_id = :id
                    """),
                    {"id": alert.alert_id},
                )
                log_row = row.fetchone()

            assert log_row[0] == "confirmed"
            assert log_row[1] is not None

        finally:
            await cleanup_alert_log(alert.alert_id)

    async def test_confirm_clears_redis_dedup_key(self, redis_client):
        """
        Confirming an alert should clear the Redis dedup key so the
        incident keeps producing fresh alerts.
        """
        alert = make_alert(camera_id="CAM_FB_REDIS_01")
        await insert_alert_log(alert)

        # Manually set a dedup key to simulate an active incident
        dedup_key = f"{alert.camera_id}:{alert.zone_id}:{alert.pipeline.value}"
        await redis_client.setex(dedup_key, 45, "0.88")

        feedback = OperatorFeedback(
            alert_id=alert.alert_id, action=OperatorAction.CONFIRM
        )

        try:
            async with db_session_scope() as session:
                result = await process_feedback(
                    feedback, alert, session, redis_client
                )

            # Key should be gone
            remaining = await redis_client.get(dedup_key)
            assert remaining is None
            assert result["dedup_key_cleared"] is True

        finally:
            await cleanup_alert_log(alert.alert_id)
            await redis_client.delete(dedup_key)


# ─────────────────────────────────────────────────────────────
# Outbox worker
# ─────────────────────────────────────────────────────────────

class TestOutboxWorker:

    async def test_worker_processes_pending_row_and_writes_to_qdrant(
        self, qdrant_client
    ):
        """
        Full outbox round-trip:
        1. Dismiss an alert → outbox row written
        2. Run outbox worker → Qdrant embedding written
        3. Verify vector exists in Qdrant
        """
        alert = make_alert(camera_id="CAM_FB_WORKER_01")
        feedback = OperatorFeedback(
            alert_id=alert.alert_id,
            action=OperatorAction.DISMISS,
            permanent=True,
        )
        point_id = _to_point_id(alert.alert_id)

        try:
            async with db_session_scope() as session:
                await process_feedback(feedback, alert, session)

            # Run the worker
            async with db_session_scope() as session:
                processed = await process_outbox_batch(
                    session=session,
                    qdrant_client=qdrant_client,
                    batch_size=10,
                )

            assert processed >= 1

            # Verify embedding is now in Qdrant
            points = await qdrant_client.retrieve(
                collection_name=settings.qdrant_collection_dismissed,
                ids=[point_id],
                with_payload=True,
            )
            assert len(points) == 1
            assert points[0].payload["alert_id"] == alert.alert_id
            assert points[0].payload["is_permanent"] is True

        finally:
            await cleanup_rule_by_camera("CAM_FB_WORKER_01")
            await cleanup_outbox_by_camera("CAM_FB_WORKER_01")
            try:
                await qdrant_client.delete(
                    collection_name=settings.qdrant_collection_dismissed,
                    points_selector=qmodels.PointIdsList(points=[point_id]),
                )
            except Exception:
                pass

    async def test_worker_marks_row_processed_after_success(self, qdrant_client):
        alert = make_alert(camera_id="CAM_FB_WORKER_02")
        feedback = OperatorFeedback(
            alert_id=alert.alert_id,
            action=OperatorAction.DISMISS,
            permanent=False,
        )
        point_id = _to_point_id(alert.alert_id)

        try:
            async with db_session_scope() as session:
                await process_feedback(feedback, alert, session)

            async with db_session_scope() as session:
                await process_outbox_batch(
                    session=session,
                    qdrant_client=qdrant_client,
                )

            # Row should now have processed_at set
            async with db_session_scope() as session:
                row = await session.execute(
                    text("""
                        SELECT processed_at FROM outbox
                        WHERE payload::jsonb->>'camera_id' = :c
                        ORDER BY created_at DESC LIMIT 1
                    """),
                    {"c": alert.camera_id},
                )
                outbox_row = row.fetchone()

            assert outbox_row[0] is not None  # processed_at is set

        finally:
            await cleanup_rule_by_camera("CAM_FB_WORKER_02")
            await cleanup_outbox_by_camera("CAM_FB_WORKER_02")
            try:
                await qdrant_client.delete(
                    collection_name=settings.qdrant_collection_dismissed,
                    points_selector=qmodels.PointIdsList(points=[point_id]),
                )
            except Exception:
                pass

    async def test_worker_empty_outbox_returns_zero(self, qdrant_client):
        """Worker should handle an empty outbox gracefully."""
        # Process anything currently pending first
        async with db_session_scope() as session:
            await process_outbox_batch(session=session, qdrant_client=qdrant_client)

        # Now run again on an empty table
        async with db_session_scope() as session:
            processed = await process_outbox_batch(
                session=session, qdrant_client=qdrant_client
            )
        assert processed == 0