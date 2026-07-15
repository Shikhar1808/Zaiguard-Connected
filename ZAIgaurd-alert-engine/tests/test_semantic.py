"""
Tests for layers/suppression/semantic.py (Layer 4B)
=====================================================
Tests description string building, embedding calculation,
collection initialization, live ANN search, TTL filtering,
and the store_dismissed_alert write path.

Docker must be running with zaiguard_qdrant healthy:
    docker compose ps
    pytest tests/test_semantic.py -v
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels

from config.settings import settings
from config.thresholds import config_loader
from layers.suppression.semantic import (
    DEFAULT_DISMISSAL_TTL_HOURS,
    SemanticEmbedder,
    SemanticRuleResult,
    _to_point_id,
    build_alert_description,
    embedder,
    ensure_qdrant_collections,
    run_semantic_suppression_check,
    store_dismissed_alert,
)
from models.schemas import AlertEvent, Pipeline, RawDetectionEvent


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def make_test_event(
    pipeline: Pipeline = Pipeline.VIOLENCE,
    camera_id: str = "CAM_TEST_SEM_01",
    zone_id: str = "gym_east",
    zone_label: str = "Gym East",
    hour: int = 17,
    day_of_week: int = 0,
    confidence: float = 0.88,
    features: dict | None = None,
) -> AlertEvent:
    raw = RawDetectionEvent(
        pipeline=pipeline,
        raw_confidence=confidence,
        camera_id=camera_id,
        zone_id=zone_id,
        zone_label=zone_label,
        timestamp=datetime(2026, 6, 8, hour, 0, 0, tzinfo=timezone.utc),
        pipeline_features=(
            features if features is not None
            else {"inter_person_distance": 0.5, "relative_velocity": 2.1}
        ),
    )
    return AlertEvent(
        alert_id=f"test_sem_{uuid.uuid4().hex[:16]}",
        source_event=raw,
        effective_conf=confidence,
        hour_of_day=hour,
        day_of_week=day_of_week,
    )


async def _delete_point(client: AsyncQdrantClient, point_id: str) -> None:
    """Helper: delete one point from dismissed_alerts, ignoring if absent."""
    try:
        await client.delete(
            collection_name=settings.qdrant_collection_dismissed,
            points_selector=qmodels.PointIdsList(points=[point_id]),
        )
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────
# Session-scoped fixtures
# ─────────────────────────────────────────────────────────────

@pytest_asyncio.fixture(scope="session")
async def qdrant_client():
    """AsyncQdrantClient connected to live local Qdrant container."""
    client = AsyncQdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
    )
    await ensure_qdrant_collections(client)
    yield client
    await client.close()


@pytest_asyncio.fixture(scope="session", autouse=True)
async def ensure_config_loaded():
    """
    Loads threshold config from Postgres before semantic tests run.
    Needed because run_semantic_suppression_check() calls
    config_loader.get_similarity_threshold() internally.
    Soft-fails if Postgres is unreachable — semantic tests don't
    strictly require live config.
    """
    try:
        await config_loader.ensure_fresh()
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────
# 1. Description-String Builder
# ─────────────────────────────────────────────────────────────

class TestDescriptionBuilder:

    def test_alert_event_format_and_feature_sorting(self):
        """
        Features must be sorted alphabetically so two events with the
        same feature values but different dict insertion order produce
        identical descriptions and therefore identical embeddings.
        """
        event = make_test_event(
            pipeline=Pipeline.VIOLENCE,
            camera_id="CAM_TEST_SEM_01",
            zone_label="Gym East",
            hour=17,
            day_of_week=0,
            features={"b_score": 1.2, "a_dist": 0.4},
        )
        desc = build_alert_description(event)
        assert desc == (
            "Pipeline: violence | Camera: CAM_TEST_SEM_01 | "
            "Zone: Gym East | Time: 17:00 on Monday | "
            "Features: a_dist=0.4, b_score=1.2"
        )

    def test_raw_detection_event_uses_timestamp(self):
        """RawDetectionEvent has no hour_of_day — must derive from timestamp."""
        raw = RawDetectionEvent(
            pipeline=Pipeline.FIRE,
            raw_confidence=0.92,
            camera_id="CAM_TEST_SEM_02",
            zone_label="Cafeteria",
            # 2026-06-09 is a Tuesday
            timestamp=datetime(2026, 6, 9, 14, 30, tzinfo=timezone.utc),
            pipeline_features={"smoke_area": 15.5},
        )
        desc = build_alert_description(raw)
        assert desc == (
            "Pipeline: fire | Camera: CAM_TEST_SEM_02 | "
            "Zone: Cafeteria | Time: 14:00 on Tuesday | "
            "Features: smoke_area=15.5"
        )

    def test_empty_features_shows_none(self):
        event = make_test_event(features={})
        assert "Features: none" in build_alert_description(event)

    def test_different_days_produce_different_descriptions(self):
        mon = make_test_event(day_of_week=0)
        sat = make_test_event(day_of_week=5)
        assert "Monday" in build_alert_description(mon)
        assert "Saturday" in build_alert_description(sat)
        assert build_alert_description(mon) != build_alert_description(sat)

    def test_same_inputs_produce_identical_description(self):
        """Determinism check — same event always produces same string."""
        e1 = make_test_event(camera_id="CAM_DET", features={"x": 1.0, "y": 2.0})
        e2 = make_test_event(camera_id="CAM_DET", features={"y": 2.0, "x": 1.0})
        assert build_alert_description(e1) == build_alert_description(e2)


# ─────────────────────────────────────────────────────────────
# 2. Embedding Manager
# ─────────────────────────────────────────────────────────────

class TestSemanticEmbedder:

    def test_produces_384_dimensional_vector(self):
        v = embedder.encode("test text")
        assert len(v) == 384

    def test_output_is_l2_normalized(self):
        v = embedder.encode(
            "Pipeline: fire | Camera: CAM_01 | Zone: Hallway | "
            "Time: 12:00 on Monday | Features: none"
        )
        norm = math.sqrt(sum(x * x for x in v))
        assert pytest.approx(norm, rel=1e-4) == 1.0

    def test_identical_text_produces_identical_vector(self):
        s = "deterministic embedding test string"
        assert embedder.encode(s) == embedder.encode(s)

    def test_different_text_produces_different_vector(self):
        v1 = embedder.encode("violence in the gym at 5pm")
        v2 = embedder.encode("fire detected in kitchen at midnight")
        assert v1 != v2


# ─────────────────────────────────────────────────────────────
# 3. Point ID Conversion
# ─────────────────────────────────────────────────────────────

class TestPointIdConversion:

    def test_32_hex_chars_become_hyphenated_uuid(self):
        assert _to_point_id("a" * 32) == "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"

    def test_non_hex_id_falls_back_to_uuid5(self):
        point_id = _to_point_id("some-custom-alert-id")
        # Must be a valid UUID string without raising
        parsed = uuid.UUID(point_id)
        assert str(parsed) == point_id

    def test_same_input_produces_same_point_id(self):
        assert _to_point_id("abc123" * 5 + "ab") == _to_point_id("abc123" * 5 + "ab")


# ─────────────────────────────────────────────────────────────
# 4. Qdrant Collection Setup
# ─────────────────────────────────────────────────────────────

class TestQdrantSetup:

    async def test_collection_exists_after_setup(
        self, qdrant_client: AsyncQdrantClient
    ):
        await ensure_qdrant_collections(qdrant_client)
        cols = await qdrant_client.get_collections()
        names = [c.name for c in cols.collections]
        assert settings.qdrant_collection_dismissed in names

    async def test_idempotent_setup_does_not_raise(
        self, qdrant_client: AsyncQdrantClient
    ):
        """Calling ensure_qdrant_collections twice should not raise."""
        await ensure_qdrant_collections(qdrant_client)
        await ensure_qdrant_collections(qdrant_client)


# ─────────────────────────────────────────────────────────────
# 5. Semantic Suppression Check — live Qdrant
# ─────────────────────────────────────────────────────────────

class TestSemanticSuppressionCheck:

    async def test_no_dismissed_alerts_passes_through(
        self, qdrant_client: AsyncQdrantClient
    ):
        event = make_test_event(camera_id="CAM_TEST_UNIQUE_EMPTY_999")
        result = await run_semantic_suppression_check(event, qdrant_client)
        assert result.suppressed is False
        assert result.nearest_score >= 0.0

    async def test_very_similar_event_is_suppressed(
        self, qdrant_client: AsyncQdrantClient
    ):
        """
        Store a dismissed alert then query with a nearly identical one.
        The similarity should exceed the violence threshold (0.88).
        """
        event1 = make_test_event(
            pipeline=Pipeline.VIOLENCE,
            camera_id="CAM_TEST_SEM_10",
            zone_label="Gym East",
            features={"inter_person_distance": 0.5, "relative_velocity": 2.1},
        )
        point_id = _to_point_id(event1.alert_id)

        try:
            await store_dismissed_alert(
                qdrant_client, event1.alert_id, event1, permanent=True
            )

            event2 = make_test_event(
                pipeline=Pipeline.VIOLENCE,
                camera_id="CAM_TEST_SEM_10",
                zone_label="Gym East",
                features={"inter_person_distance": 0.51, "relative_velocity": 2.05},
            )

            result = await run_semantic_suppression_check(event2, qdrant_client)

            assert result.suppressed is True
            assert result.nearest_alert_id == event1.alert_id
            assert result.nearest_score > 0.90

        finally:
            await _delete_point(qdrant_client, point_id)

    async def test_different_pipeline_not_suppressed(
        self, qdrant_client: AsyncQdrantClient
    ):
        """
        Payload filter on pipeline must prevent a dismissed fire alert
        from suppressing a violence alert, even with similar descriptions.
        """
        fire_event = make_test_event(
            pipeline=Pipeline.FIRE,
            camera_id="CAM_TEST_SEM_20",
            zone_label="Lobby",
            features={"smoke_density": 0.8},
        )
        point_id = _to_point_id(fire_event.alert_id)

        try:
            await store_dismissed_alert(
                qdrant_client, fire_event.alert_id, fire_event, permanent=True
            )

            violence_event = make_test_event(
                pipeline=Pipeline.VIOLENCE,
                camera_id="CAM_TEST_SEM_20",
                zone_label="Lobby",
                features={"smoke_density": 0.8},
            )

            result = await run_semantic_suppression_check(
                violence_event, qdrant_client
            )
            assert result.suppressed is False

        finally:
            await _delete_point(qdrant_client, point_id)

    async def test_result_dataclass_is_frozen(
        self, qdrant_client: AsyncQdrantClient
    ):
        """SemanticRuleResult must be immutable — frozen=True."""
        event = make_test_event()
        result = await run_semantic_suppression_check(event, qdrant_client)
        with pytest.raises((AttributeError, TypeError)):
            result.suppressed = True  # type: ignore


# ─────────────────────────────────────────────────────────────
# 6. TTL Filtering — the gap closed in this revision
# ─────────────────────────────────────────────────────────────

class TestTTLFiltering:
    """
    Verifies that expired non-permanent dismissals are excluded from
    ANN search, and permanent ones are always included.

    We control TTL by writing the payload directly rather than using
    store_dismissed_alert's default 24h window — this lets us insert
    already-expired records without sleeping.
    """

    async def _insert_with_custom_ttl(
        self,
        client: AsyncQdrantClient,
        event: AlertEvent,
        is_permanent: bool,
        ttl_expires: str | None,
    ) -> str:
        """Insert a point with manually specified TTL fields."""
        description = build_alert_description(event)
        vector = embedder.encode(description)
        point_id = _to_point_id(event.alert_id)

        await client.upsert(
            collection_name=settings.qdrant_collection_dismissed,
            points=[
                qmodels.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "alert_id":    event.alert_id,
                        "pipeline":    event.pipeline.value,
                        "camera_id":   event.camera_id,
                        "zone_id":     event.zone_id,
                        "zone_label":  event.zone_label,
                        "description": description,
                        "is_permanent": is_permanent,
                        "ttl_expires":  ttl_expires,
                    },
                )
            ],
        )
        return point_id

    async def test_permanent_dismissal_suppresses(
        self, qdrant_client: AsyncQdrantClient
    ):
        """is_permanent=True should always suppress regardless of ttl_expires."""
        event = make_test_event(
            camera_id="CAM_TTL_PERM",
            features={"inter_person_distance": 0.5, "relative_velocity": 2.1},
        )
        point_id = await self._insert_with_custom_ttl(
            qdrant_client, event,
            is_permanent=True,
            ttl_expires=None,
        )

        try:
            query = make_test_event(
                camera_id="CAM_TTL_PERM",
                features={"inter_person_distance": 0.51, "relative_velocity": 2.09},
            )
            result = await run_semantic_suppression_check(query, qdrant_client)
            assert result.suppressed is True
        finally:
            await _delete_point(qdrant_client, point_id)

    async def test_non_expired_dismissal_suppresses(
        self, qdrant_client: AsyncQdrantClient
    ):
        """ttl_expires in the future → still active, should suppress."""
        future_ttl = (
            datetime.now(timezone.utc) + timedelta(hours=23)
        ).isoformat()

        event = make_test_event(
            camera_id="CAM_TTL_FUTURE",
            features={"inter_person_distance": 0.5, "relative_velocity": 2.1},
        )
        point_id = await self._insert_with_custom_ttl(
            qdrant_client, event,
            is_permanent=False,
            ttl_expires=future_ttl,
        )

        try:
            query = make_test_event(
                camera_id="CAM_TTL_FUTURE",
                features={"inter_person_distance": 0.51, "relative_velocity": 2.09},
            )
            result = await run_semantic_suppression_check(query, qdrant_client)
            assert result.suppressed is True
        finally:
            await _delete_point(qdrant_client, point_id)

    async def test_expired_dismissal_does_not_suppress(
        self, qdrant_client: AsyncQdrantClient
    ):
        """
        ttl_expires in the past AND is_permanent=False → expired.
        The query should NOT match this point, so the event passes through.
        """
        past_ttl = (
            datetime.now(timezone.utc) - timedelta(hours=1)
        ).isoformat()

        event = make_test_event(
            camera_id="CAM_TTL_EXPIRED",
            features={"inter_person_distance": 0.5, "relative_velocity": 2.1},
        )
        point_id = await self._insert_with_custom_ttl(
            qdrant_client, event,
            is_permanent=False,
            ttl_expires=past_ttl,
        )

        try:
            query = make_test_event(
                camera_id="CAM_TTL_EXPIRED",
                features={"inter_person_distance": 0.51, "relative_velocity": 2.09},
            )
            result = await run_semantic_suppression_check(query, qdrant_client)
            assert result.suppressed is False
        finally:
            await _delete_point(qdrant_client, point_id)


# ─────────────────────────────────────────────────────────────
# 7. store_dismissed_alert write path
# ─────────────────────────────────────────────────────────────

class TestStoreDismissedAlert:

    async def test_non_permanent_stores_ttl_expires(
        self, qdrant_client: AsyncQdrantClient
    ):
        """
        A non-permanent dismissal must write ttl_expires and
        is_permanent=False to the payload.
        """
        event = make_test_event(camera_id="CAM_STORE_NONPERM")
        point_id = _to_point_id(event.alert_id)

        try:
            await store_dismissed_alert(
                qdrant_client, event.alert_id, event, permanent=False
            )

            points = await qdrant_client.retrieve(
                collection_name=settings.qdrant_collection_dismissed,
                ids=[point_id],
                with_payload=True,
            )
            assert len(points) == 1
            payload = points[0].payload
            assert payload["is_permanent"] is False
            assert payload["ttl_expires"] is not None

            # ttl_expires should be approximately 24h from now
            ttl_dt = datetime.fromisoformat(payload["ttl_expires"])
            diff = ttl_dt - datetime.now(timezone.utc)
            assert timedelta(hours=23) < diff < timedelta(hours=25)

        finally:
            await _delete_point(qdrant_client, point_id)

    async def test_permanent_stores_is_permanent_true(
        self, qdrant_client: AsyncQdrantClient
    ):
        """A permanent dismissal must write is_permanent=True and no ttl_expires."""
        event = make_test_event(camera_id="CAM_STORE_PERM")
        point_id = _to_point_id(event.alert_id)

        try:
            await store_dismissed_alert(
                qdrant_client, event.alert_id, event, permanent=True
            )

            points = await qdrant_client.retrieve(
                collection_name=settings.qdrant_collection_dismissed,
                ids=[point_id],
                with_payload=True,
            )
            assert len(points) == 1
            payload = points[0].payload
            assert payload["is_permanent"] is True
            assert payload["ttl_expires"] is None

        finally:
            await _delete_point(qdrant_client, point_id)

    async def test_stored_point_can_be_retrieved(
        self, qdrant_client: AsyncQdrantClient
    ):
        """Full round-trip: store a dismissed alert, verify it exists in Qdrant."""
        event = make_test_event(camera_id="CAM_STORE_RETRIEVE")
        point_id = _to_point_id(event.alert_id)

        try:
            await store_dismissed_alert(
                qdrant_client, event.alert_id, event, permanent=True
            )

            points = await qdrant_client.retrieve(
                collection_name=settings.qdrant_collection_dismissed,
                ids=[point_id],
                with_payload=True,
            )
            assert len(points) == 1
            assert points[0].payload["alert_id"] == event.alert_id
            assert points[0].payload["pipeline"] == "violence"
            assert points[0].payload["camera_id"] == "CAM_STORE_RETRIEVE"

        finally:
            await _delete_point(qdrant_client, point_id)