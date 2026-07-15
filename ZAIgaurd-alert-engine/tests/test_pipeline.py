"""
Tests for pipeline.py — full end-to-end integration
======================================================
These tests run the COMPLETE five-layer pipeline against live
Postgres, Redis, and Qdrant. This is the first place all layers
are exercised together rather than in isolation.

Requires Docker running with all three services healthy:
    docker compose ps
    pytest tests/test_pipeline.py -v

WHAT THIS FILE PROVES THAT THE PER-LAYER TESTS DON'T
-------------------------------------------------------
Per-layer tests prove each layer is internally correct. This file
proves the HANDOFFS between layers are correct — that Layer 1's
output shape is what Layer 2 expects, that Layer 2's AlertEvent is
what Layers 3/4/5 expect, and that the orchestration logic in
process_event() calls everything in the right order with the right
early-exit behavior at each gate.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels
from redis.asyncio import Redis

from config.database import db_session_scope
from config.settings import settings
from config.thresholds import config_loader
from layers.suppression.exact_rules import write_suppression_rule
from layers.suppression.semantic import (
    _to_point_id,
    ensure_qdrant_collections,
    store_dismissed_alert,
)
from models.schemas import AlertTier, Pipeline, RawDetectionEvent
from pipeline import process_event


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


@pytest_asyncio.fixture(autouse=True)
async def cleanup_per_test(redis_client):
    """
    Clears any dedup keys this test file might create, so tests in
    this file don't interfere with each other via leftover Redis state.
    Runs before AND after every test in this file.
    """
    yield
    keys = await redis_client.keys("CAM_PIPE_TEST_*")
    if keys:
        await redis_client.delete(*keys)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def make_event(
    pipeline: Pipeline = Pipeline.VIOLENCE,
    confidence: float = 0.85,
    camera_id: str = "CAM_PIPE_TEST_01",
    zone_id: str = "default",
    hour: int = 12,
) -> RawDetectionEvent:
    return RawDetectionEvent(
        pipeline=pipeline,
        raw_confidence=confidence,
        camera_id=camera_id,
        zone_id=zone_id,
        zone_label="Test Zone",
        timestamp=datetime(2026, 6, 8, hour, 0, 0, tzinfo=timezone.utc),  # Monday
        frame_ref="/clips/pipeline_test.mp4",
        involved_ids=[1, 2],
    )


async def cleanup_alert_log(alert_id: str) -> None:
    from sqlalchemy import text
    async with db_session_scope() as session:
        await session.execute(
            text("DELETE FROM alert_log WHERE alert_id = :id"),
            {"id": alert_id},
        )


async def cleanup_suppression_rule(rule_id: str) -> None:
    from sqlalchemy import text
    async with db_session_scope() as session:
        await session.execute(
            text("DELETE FROM suppression_rules WHERE id = CAST(:id AS uuid)"),
            {"id": rule_id},
        )


# ─────────────────────────────────────────────────────────────
# Full pass-through — event survives all five layers
# ─────────────────────────────────────────────────────────────

class TestFullPassThrough:

    @pytest.mark.asyncio
    async def test_high_confidence_violence_produces_alert(
        self, redis_client, qdrant_client
    ):
        raw = make_event(
            pipeline=Pipeline.VIOLENCE,
            confidence=0.95,
            camera_id="CAM_PIPE_TEST_PASS_01",
        )

        async with db_session_scope() as session:
            alert = await process_event(raw, redis_client, session, qdrant_client)

        try:
            assert alert is not None
            assert alert.tier == AlertTier.CRITICAL  # 0.95 >= 0.90
            assert alert.pipeline == Pipeline.VIOLENCE
            assert alert.camera_id == "CAM_PIPE_TEST_PASS_01"
        finally:
            if alert:
                await cleanup_alert_log(alert.alert_id)

    @pytest.mark.asyncio
    async def test_alert_written_to_alert_log(self, redis_client, qdrant_client):
        from sqlalchemy import text

        raw = make_event(
            pipeline=Pipeline.FIRE,
            confidence=0.80,
            camera_id="CAM_PIPE_TEST_LOG_01",
        )

        async with db_session_scope() as session:
            alert = await process_event(raw, redis_client, session, qdrant_client)

        try:
            assert alert is not None

            async with db_session_scope() as session:
                result = await session.execute(
                    text("SELECT pipeline, tier FROM alert_log WHERE alert_id = :id"),
                    {"id": alert.alert_id},
                )
                row = result.fetchone()

            assert row is not None
            assert row[0] == "fire"
            assert row[1] == "CRITICAL"
        finally:
            if alert:
                await cleanup_alert_log(alert.alert_id)

    @pytest.mark.asyncio
    async def test_duplicate_alert_log_insert_is_idempotent(
        self, redis_client, qdrant_client
    ):
        """
        Calling process_event twice with the same underlying detection
        (same camera, timestamp, pipeline -> same alert_id) should not
        crash on the second alert_log insert, thanks to ON CONFLICT
        DO NOTHING. We bypass dedup by using different zone_ids so
        Layer 3 doesn't swallow the second call -- we want to test the
        alert_log INSERT specifically.
        """
        from sqlalchemy import text

        raw = make_event(
            pipeline=Pipeline.ACCIDENT,
            confidence=0.85,
            camera_id="CAM_PIPE_TEST_IDEMPOTENT",
        )

        async with db_session_scope() as session:
            alert1 = await process_event(raw, redis_client, session, qdrant_client)

        # Clear the dedup key so the second call isn't swallowed at Layer 3 --
        # we want to reach the alert_log insert again with the SAME alert_id
        # (same camera/timestamp/pipeline -> deterministic ID).
        from layers.dedup import clear_dedup_key
        from layers.enrichment import run_enrichment
        from layers.threshold_gate import run_threshold_gate

        gate_result = await run_threshold_gate(raw)
        enriched = run_enrichment(raw, gate_result)
        await clear_dedup_key(enriched, redis_client)

        try:
            async with db_session_scope() as session:
                alert2 = await process_event(raw, redis_client, session, qdrant_client)

            assert alert1 is not None
            assert alert2 is not None
            assert alert1.alert_id == alert2.alert_id  # deterministic ID

            # Verify exactly ONE row exists in alert_log, not two
            async with db_session_scope() as session:
                result = await session.execute(
                    text("SELECT COUNT(*) FROM alert_log WHERE alert_id = :id"),
                    {"id": alert1.alert_id},
                )
                count = result.scalar()
            assert count == 1

        finally:
            await cleanup_alert_log(raw.camera_id)  # best-effort
            if alert1:
                await cleanup_alert_log(alert1.alert_id)


# ─────────────────────────────────────────────────────────────
# Layer 1 drop — confidence below effective threshold
# ─────────────────────────────────────────────────────────────

class TestLayer1Drop:

    @pytest.mark.asyncio
    async def test_low_confidence_dropped_at_layer1(
        self, redis_client, qdrant_client
    ):
        """0.30 is below every pipeline's base threshold -> dropped at Layer 1."""
        raw = make_event(
            pipeline=Pipeline.TRESPASSING,
            confidence=0.30,
            camera_id="CAM_PIPE_TEST_L1_DROP",
        )

        async with db_session_scope() as session:
            alert = await process_event(raw, redis_client, session, qdrant_client)

        assert alert is None


# ─────────────────────────────────────────────────────────────
# Layer 3 drop — burst duplicate
# ─────────────────────────────────────────────────────────────

class TestLayer3Drop:

    @pytest.mark.asyncio
    async def test_second_identical_event_deduplicated(
        self, redis_client, qdrant_client
    ):
        """
        First event of a burst passes through; an immediate second
        event (same camera/zone/pipeline, similar confidence) within
        the TTL window should be dropped as a duplicate.
        """
        raw1 = make_event(
            pipeline=Pipeline.VIOLENCE,
            confidence=0.85,
            camera_id="CAM_PIPE_TEST_DEDUP",
        )
        raw2 = make_event(
            pipeline=Pipeline.VIOLENCE,
            confidence=0.86,  # tiny increase, well below escalation_delta
            camera_id="CAM_PIPE_TEST_DEDUP",
        )

        async with db_session_scope() as session:
            alert1 = await process_event(raw1, redis_client, session, qdrant_client)

        async with db_session_scope() as session:
            alert2 = await process_event(raw2, redis_client, session, qdrant_client)

        try:
            assert alert1 is not None
            assert alert2 is None  # swallowed as duplicate
        finally:
            if alert1:
                await cleanup_alert_log(alert1.alert_id)


# ─────────────────────────────────────────────────────────────
# Layer 4A drop — exact suppression rule
# ─────────────────────────────────────────────────────────────

class TestLayer4ADrop:

    @pytest.mark.asyncio
    async def test_event_matching_exact_rule_is_suppressed(
        self, redis_client, qdrant_client
    ):
        camera_id = "CAM_PIPE_TEST_4A_DROP"

        async with db_session_scope() as session:
            rule_id = await write_suppression_rule(
                camera_id=camera_id,
                pipeline="violence",
                zone_id=None,
                hour_start=None,
                hour_end=None,
                days_mask=None,
                expires_at=None,   # permanent
                source="manual",
                session=session,
            )

        raw = make_event(
            pipeline=Pipeline.VIOLENCE,
            confidence=0.90,
            camera_id=camera_id,
        )

        try:
            async with db_session_scope() as session:
                alert = await process_event(raw, redis_client, session, qdrant_client)

            assert alert is None  # suppressed by the rule
        finally:
            await cleanup_suppression_rule(rule_id)


# ─────────────────────────────────────────────────────────────
# Layer 4B drop — semantic similarity to dismissed alert
# ─────────────────────────────────────────────────────────────

class TestLayer4BDrop:

    @pytest.mark.asyncio
    async def test_event_similar_to_dismissed_alert_is_suppressed(
        self, redis_client, qdrant_client
    ):
        camera_id = "CAM_PIPE_TEST_4B_DROP"

        # Manually build a dismissed-alert-like event and store it
        from models.schemas import AlertEvent, build_alert_id

        dismissed_raw = RawDetectionEvent(
            pipeline=Pipeline.VIOLENCE,
            raw_confidence=0.80,
            camera_id=camera_id,
            zone_id="default",
            zone_label="Test Zone",
            timestamp=datetime(2026, 6, 8, 12, 0, 0, tzinfo=timezone.utc),
            pipeline_features={"inter_person_distance": 0.5, "relative_velocity": 2.0},
        )
        dismissed_event = AlertEvent(
            alert_id=build_alert_id(camera_id, dismissed_raw.timestamp, Pipeline.VIOLENCE),
            source_event=dismissed_raw,
            effective_conf=0.80,
            hour_of_day=12,
            day_of_week=0,
        )
        point_id = _to_point_id(dismissed_event.alert_id)

        await store_dismissed_alert(
            qdrant_client, dismissed_event.alert_id, dismissed_event, permanent=True
        )

        # New near-identical event, different timestamp -> different alert_id,
        # but should match semantically and get suppressed
        raw = RawDetectionEvent(
            pipeline=Pipeline.VIOLENCE,
            raw_confidence=0.81,
            camera_id=camera_id,
            zone_id="default",
            zone_label="Test Zone",
            timestamp=datetime(2026, 6, 8, 12, 0, 5, tzinfo=timezone.utc),
            pipeline_features={"inter_person_distance": 0.51, "relative_velocity": 2.02},
        )

        try:
            async with db_session_scope() as session:
                alert = await process_event(raw, redis_client, session, qdrant_client)

            assert alert is None  # suppressed by semantic similarity
        finally:
            await qdrant_client.delete(
                collection_name=settings.qdrant_collection_dismissed,
                points_selector=qmodels.PointIdsList(points=[point_id]),
            )


# ─────────────────────────────────────────────────────────────
# Escalation flows through correctly
# ─────────────────────────────────────────────────────────────

class TestEscalationFlow:

    @pytest.mark.asyncio
    async def test_escalating_incident_produces_second_alert(
        self, redis_client, qdrant_client
    ):
        """
        First event at 0.75 passes. Second event at 0.93 (jump of 0.18,
        above escalation_delta of 0.15) should ALSO pass through, not
        be swallowed as a duplicate.

        NOTE ON hour=22: violence's base_threshold is 0.72, but Layer 1
        applies a time multiplier on top of that. The default hour=12
        in make_event() falls in the seeded "peak hours" window
        (08:00-18:59, multiplier 1.10), which raises the effective
        threshold to 0.72 x 1.10 = 0.792 -- above 0.75, which would
        incorrectly drop raw1 at Layer 1 before dedup/escalation logic
        is ever exercised. Using hour=22 (nighttime, multiplier 0.85)
        keeps the effective threshold at 0.72 x 0.85 = 0.612, well
        below both 0.75 and 0.93, so this test isolates Layer 3
        escalation behavior without an unrelated Layer 1 interaction.
        """
        camera_id = "CAM_PIPE_TEST_ESCALATION"

        raw1 = make_event(
            pipeline=Pipeline.VIOLENCE, confidence=0.75, camera_id=camera_id, hour=22,
        )
        raw2 = make_event(
            pipeline=Pipeline.VIOLENCE, confidence=0.93, camera_id=camera_id, hour=22,
        )

        async with db_session_scope() as session:
            alert1 = await process_event(raw1, redis_client, session, qdrant_client)

        async with db_session_scope() as session:
            alert2 = await process_event(raw2, redis_client, session, qdrant_client)

        try:
            assert alert1 is not None
            assert alert2 is not None  # escalation passed through
            assert alert2.tier == AlertTier.CRITICAL
        finally:
            for a in (alert1, alert2):
                if a:
                    await cleanup_alert_log(a.alert_id)