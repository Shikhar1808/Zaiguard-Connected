"""
Tests for layers/dedup.py
===========================
Uses fakeredis — an in-memory Redis implementation — so these tests
run without a live Redis instance. fakeredis behaves identically to
real Redis for every operation we use (GET, SETEX, DEL, TTL), making
it the right tool for unit testing Redis-dependent logic.

The one thing fakeredis CAN'T fully test is real-time TTL expiry
(waiting 45 real seconds for a key to expire). Instead we verify
that TTLs are set to the correct value and trust Redis's own TTL
implementation. Integration tests against real Redis (run as part
of the full stack test in Step 9) verify actual expiry behavior.

Run with:
    pytest tests/test_dedup.py -v
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest
import fakeredis.aioredis as fakeredis

from config.thresholds import ThresholdConfigCache, ThresholdConfigLoader
from layers.dedup import DedupResult, build_dedup_key, clear_dedup_key, run_dedup
from models.schemas import AlertEvent, Pipeline, RawDetectionEvent


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def make_loader(
    violence_ttl: int = 45,
    fire_ttl: int = 120,
    escalation_delta: float = 0.15,
) -> ThresholdConfigLoader:
    loader = ThresholdConfigLoader()
    cache = ThresholdConfigCache()
    cache.thresholds = {
        "fire":        (0.60, fire_ttl,     escalation_delta),
        "violence":    (0.72, violence_ttl, escalation_delta),
        "dog_attack":  (0.68, 45,           escalation_delta),
        "trespassing": (0.78, 60,           escalation_delta),
        "accident":    (0.70, 60,           escalation_delta),
    }
    cache.time_multipliers = []
    cache.zone_multipliers = {}
    cache.tier_rules = {}
    cache.similarity_thresholds = {}
    cache.loaded_at = time.monotonic()
    loader._cache = cache
    return loader


def make_alert_event(
    pipeline: Pipeline = Pipeline.VIOLENCE,
    confidence: float = 0.85,
    camera_id: str = "CAM_07",
    zone_id: str = "gym_east",
    hour: int = 12,
) -> AlertEvent:
    raw = RawDetectionEvent(
        pipeline=pipeline,
        raw_confidence=confidence,
        camera_id=camera_id,
        zone_id=zone_id,
        zone_label="Gym East",
        timestamp=datetime(2026, 6, 8, hour, 0, 0, tzinfo=timezone.utc),
    )
    return AlertEvent(
        alert_id=f"test_{camera_id}_{pipeline.value}_{hour}",
        source_event=raw,
        effective_conf=confidence,
        hour_of_day=hour,
        day_of_week=0,
    )


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def redis():
    """Fresh in-memory fakeredis instance per test."""
    return fakeredis.FakeRedis(decode_responses=True)


@pytest.fixture
def loader():
    return make_loader()


# ─────────────────────────────────────────────────────────────
# build_dedup_key
# ─────────────────────────────────────────────────────────────

class TestBuildDedupKey:

    def test_key_format(self):
        event = make_alert_event(
            camera_id="CAM_07",
            zone_id="gym_east",
            pipeline=Pipeline.VIOLENCE,
        )
        assert build_dedup_key(event) == "CAM_07:gym_east:violence"

    def test_different_cameras_produce_different_keys(self):
        e1 = make_alert_event(camera_id="CAM_01")
        e2 = make_alert_event(camera_id="CAM_02")
        assert build_dedup_key(e1) != build_dedup_key(e2)

    def test_different_zones_produce_different_keys(self):
        e1 = make_alert_event(zone_id="gym_east")
        e2 = make_alert_event(zone_id="parking_lot")
        assert build_dedup_key(e1) != build_dedup_key(e2)

    def test_different_pipelines_produce_different_keys(self):
        e1 = make_alert_event(pipeline=Pipeline.VIOLENCE)
        e2 = make_alert_event(pipeline=Pipeline.FIRE)
        assert build_dedup_key(e1) != build_dedup_key(e2)

    def test_same_inputs_same_key(self):
        e1 = make_alert_event()
        e2 = make_alert_event()
        assert build_dedup_key(e1) == build_dedup_key(e2)

    def test_all_five_pipelines_produce_valid_keys(self):
        for pipeline in Pipeline:
            event = make_alert_event(pipeline=pipeline)
            key = build_dedup_key(event)
            assert pipeline.value in key
            assert "CAM_07" in key


# ─────────────────────────────────────────────────────────────
# New incident — first event of this type from this camera
# ─────────────────────────────────────────────────────────────

class TestNewIncident:

    @pytest.mark.asyncio
    async def test_first_event_is_not_duplicate(self, redis, loader):
        event = make_alert_event(confidence=0.85)
        result = await run_dedup(event, redis, loader)

        assert result.is_duplicate is False
        assert result.is_escalation is False

    @pytest.mark.asyncio
    async def test_first_event_stores_confidence_in_redis(self, redis, loader):
        event = make_alert_event(confidence=0.85)
        result = await run_dedup(event, redis, loader)

        stored = await redis.get(result.dedup_key)
        assert stored is not None
        assert float(stored) == pytest.approx(0.85)

    @pytest.mark.asyncio
    async def test_stored_confidence_matches_result(self, redis, loader):
        event = make_alert_event(confidence=0.85)
        result = await run_dedup(event, redis, loader)

        assert result.stored_confidence == pytest.approx(0.85)

    @pytest.mark.asyncio
    async def test_ttl_is_set_correctly_for_violence(self, redis, loader):
        """
        Violence TTL is 45s — verify Redis has that TTL set.
        We check the TTL value rather than waiting 45 seconds.
        """
        event = make_alert_event(pipeline=Pipeline.VIOLENCE)
        result = await run_dedup(event, redis, loader)

        ttl = await redis.ttl(result.dedup_key)
        # TTL should be close to 45 — allow a 2-second window for
        # the tiny amount of time between setex and our ttl check
        assert 43 <= ttl <= 45

    @pytest.mark.asyncio
    async def test_ttl_is_set_correctly_for_fire(self, redis, loader):
        """Fire TTL is 120s — longer because fires don't resolve quickly."""
        event = make_alert_event(pipeline=Pipeline.FIRE, confidence=0.75)
        result = await run_dedup(event, redis, loader)

        ttl = await redis.ttl(result.dedup_key)
        assert 118 <= ttl <= 120

    @pytest.mark.asyncio
    async def test_dedup_key_in_result(self, redis, loader):
        event = make_alert_event(camera_id="CAM_07", zone_id="gym_east")
        result = await run_dedup(event, redis, loader)
        assert result.dedup_key == "CAM_07:gym_east:violence"


# ─────────────────────────────────────────────────────────────
# Duplicate detection — second event within TTL window
# ─────────────────────────────────────────────────────────────

class TestDuplicate:

    @pytest.mark.asyncio
    async def test_second_event_same_confidence_is_duplicate(self, redis, loader):
        event = make_alert_event(confidence=0.85)
        await run_dedup(event, redis, loader)          # first: new incident
        result = await run_dedup(event, redis, loader) # second: duplicate

        assert result.is_duplicate is True
        assert result.is_escalation is False

    @pytest.mark.asyncio
    async def test_slight_confidence_increase_still_duplicate(self, redis, loader):
        """
        A confidence jump of less than escalation_delta (0.15) is
        NOT an escalation — it's just natural detection variance on
        the same ongoing incident.
        0.85 → 0.89 = jump of 0.04, below delta of 0.15 → duplicate
        """
        first = make_alert_event(confidence=0.85)
        second = make_alert_event(confidence=0.89)

        await run_dedup(first, redis, loader)
        result = await run_dedup(second, redis, loader)

        assert result.is_duplicate is True

    @pytest.mark.asyncio
    async def test_confidence_decrease_is_duplicate(self, redis, loader):
        """
        Confidence dropping mid-incident (model becomes less certain)
        is definitely not an escalation — still a duplicate.
        """
        first = make_alert_event(confidence=0.90)
        second = make_alert_event(confidence=0.75)

        await run_dedup(first, redis, loader)
        result = await run_dedup(second, redis, loader)

        assert result.is_duplicate is True

    @pytest.mark.asyncio
    async def test_stored_confidence_unchanged_on_duplicate(self, redis, loader):
        """
        When a duplicate is dropped, the stored confidence in Redis
        should remain the ORIGINAL value, not get overwritten with
        the new lower/similar value.
        """
        first = make_alert_event(confidence=0.85)
        second = make_alert_event(confidence=0.80)

        await run_dedup(first, redis, loader)
        result = await run_dedup(second, redis, loader)

        assert result.stored_confidence == pytest.approx(0.85)

    @pytest.mark.asyncio
    async def test_many_duplicates_all_dropped(self, redis, loader):
        """Simulate a burst of 10 near-identical detections."""
        first = make_alert_event(confidence=0.85)
        await run_dedup(first, redis, loader)

        for i in range(9):
            dupe = make_alert_event(confidence=0.85 + i * 0.005)  # tiny increments
            result = await run_dedup(dupe, redis, loader)
            assert result.is_duplicate is True, f"Event {i+2} should be duplicate"


# ─────────────────────────────────────────────────────────────
# Escalation detection — significant confidence jump
# ─────────────────────────────────────────────────────────────

class TestEscalation:

    @pytest.mark.asyncio
    async def test_large_confidence_jump_is_escalation(self, redis, loader):
        """
        0.75 → 0.92 = jump of 0.17, above delta of 0.15 → escalation
        """
        first = make_alert_event(confidence=0.75)
        second = make_alert_event(confidence=0.92)

        await run_dedup(first, redis, loader)
        result = await run_dedup(second, redis, loader)

        assert result.is_duplicate is False
        assert result.is_escalation is True

    @pytest.mark.asyncio
    async def test_escalation_boundary_exact_delta(self, redis, loader):
        """
        A jump of exactly escalation_delta (0.15) should trigger
        escalation — the check is >= not >.
        """
        first = make_alert_event(confidence=0.75)
        second = make_alert_event(confidence=0.75 + 0.15)  # exactly 0.90

        await run_dedup(first, redis, loader)
        result = await run_dedup(second, redis, loader)

        assert result.is_escalation is True

    @pytest.mark.asyncio
    async def test_just_below_escalation_delta_is_duplicate(self, redis, loader):
        """
        A jump of 0.149 (just below 0.15) should NOT escalate.
        Confirms the boundary is >= not >.
        """
        first = make_alert_event(confidence=0.75)
        second = make_alert_event(confidence=0.75 + 0.149)

        await run_dedup(first, redis, loader)
        result = await run_dedup(second, redis, loader)

        assert result.is_duplicate is True
        assert result.is_escalation is False

    @pytest.mark.asyncio
    async def test_escalation_updates_stored_confidence(self, redis, loader):
        """
        After an escalation, the stored confidence should be updated
        to the new higher value so the NEXT escalation is measured
        from the new baseline, not the original.
        """
        first = make_alert_event(confidence=0.75)
        second = make_alert_event(confidence=0.92)

        await run_dedup(first, redis, loader)
        result = await run_dedup(second, redis, loader)

        assert result.stored_confidence == pytest.approx(0.92)
        stored_in_redis = await redis.get(result.dedup_key)
        assert float(stored_in_redis) == pytest.approx(0.92)

    @pytest.mark.asyncio
    async def test_escalation_resets_ttl(self, redis, loader):
        """
        An escalating incident should get a fresh TTL window.
        Without this, an incident that escalates just before the
        original TTL expires would immediately stop deduplicating.
        """
        first = make_alert_event(confidence=0.75)
        await run_dedup(first, redis, loader)

        # Simulate time passing by manually reducing the TTL
        dedup_key = build_dedup_key(first)
        await redis.expire(dedup_key, 5)  # only 5 seconds left

        # Now an escalation arrives
        second = make_alert_event(confidence=0.92)
        result = await run_dedup(second, redis, loader)

        assert result.is_escalation is True
        # TTL should be reset to the full window, not 5
        new_ttl = await redis.ttl(dedup_key)
        assert new_ttl > 40   # reset to ~45, not stuck at 5

    @pytest.mark.asyncio
    async def test_double_escalation(self, redis, loader):
        """
        An incident escalates twice — each escalation is measured
        from the previous escalation's confidence, not the original.

        0.75 → 0.92 (jump 0.17) → escalation
        0.92 → 0.95 (jump 0.03) → duplicate (below delta)
        0.92 → 1.00 (jump 0.08 from current 0.92) → wait... 0.08 < 0.15 → still duplicate

        The key insight: after the first escalation, the stored
        confidence is 0.92. The next escalation needs to be 0.15
        above THAT — i.e. >= 1.07, which is impossible. This is
        intentional: once an incident is near maximum confidence,
        further escalations are meaningless.

        But if the first event was 0.70 and escalation raised it to
        0.85, then 0.85 + 0.15 = 1.00 is still reachable.
        """
        first = make_alert_event(confidence=0.70)
        await run_dedup(first, redis, loader)

        second = make_alert_event(confidence=0.85)  # jump 0.15 → escalation
        result_2 = await run_dedup(second, redis, loader)
        assert result_2.is_escalation is True
        assert result_2.stored_confidence == pytest.approx(0.85)

        third = make_alert_event(confidence=1.00)   # jump 0.15 from 0.85 → escalation
        result_3 = await run_dedup(third, redis, loader)
        assert result_3.is_escalation is True
        assert result_3.stored_confidence == pytest.approx(1.00)


# ─────────────────────────────────────────────────────────────
# Isolation — different camera/zone/pipeline combinations
# ─────────────────────────────────────────────────────────────

class TestIsolation:
    """
    Dedup keys are scoped to (camera, zone, pipeline).
    Events from different cameras/zones/pipelines must never
    interfere with each other.
    """

    @pytest.mark.asyncio
    async def test_different_cameras_independent(self, redis, loader):
        e1 = make_alert_event(camera_id="CAM_01")
        e2 = make_alert_event(camera_id="CAM_02")

        await run_dedup(e1, redis, loader)  # seeds CAM_01

        # CAM_02 hasn't been seen before — should be a new incident
        result = await run_dedup(e2, redis, loader)
        assert result.is_duplicate is False

    @pytest.mark.asyncio
    async def test_different_zones_independent(self, redis, loader):
        e1 = make_alert_event(zone_id="gym_east")
        e2 = make_alert_event(zone_id="parking_lot")

        await run_dedup(e1, redis, loader)
        result = await run_dedup(e2, redis, loader)
        assert result.is_duplicate is False

    @pytest.mark.asyncio
    async def test_different_pipelines_independent(self, redis, loader):
        """
        Violence and fire from the same camera are separate events —
        a violence alert should not suppress a fire alert.
        """
        violence = make_alert_event(pipeline=Pipeline.VIOLENCE)
        fire = make_alert_event(pipeline=Pipeline.FIRE, confidence=0.75)

        await run_dedup(violence, redis, loader)
        result = await run_dedup(fire, redis, loader)
        assert result.is_duplicate is False

    @pytest.mark.asyncio
    async def test_multiple_concurrent_incidents(self, redis, loader):
        """
        Simulate two simultaneous incidents on two cameras.
        Each should independently track its own dedup state.
        """
        cam1_first = make_alert_event(camera_id="CAM_01", confidence=0.80)
        cam2_first = make_alert_event(camera_id="CAM_02", confidence=0.80)

        await run_dedup(cam1_first, redis, loader)
        await run_dedup(cam2_first, redis, loader)

        # Duplicates on each camera
        cam1_dupe = make_alert_event(camera_id="CAM_01", confidence=0.81)
        cam2_dupe = make_alert_event(camera_id="CAM_02", confidence=0.81)

        r1 = await run_dedup(cam1_dupe, redis, loader)
        r2 = await run_dedup(cam2_dupe, redis, loader)

        assert r1.is_duplicate is True
        assert r2.is_duplicate is True

        # Escalation on CAM_01 should not affect CAM_02
        cam1_escalation = make_alert_event(camera_id="CAM_01", confidence=0.97)
        r3 = await run_dedup(cam1_escalation, redis, loader)
        assert r3.is_escalation is True

        cam2_normal_dupe = make_alert_event(camera_id="CAM_02", confidence=0.82)
        r4 = await run_dedup(cam2_normal_dupe, redis, loader)
        assert r4.is_duplicate is True  # CAM_02 unaffected by CAM_01 escalation


# ─────────────────────────────────────────────────────────────
# After TTL expiry — new incident window opens
# ─────────────────────────────────────────────────────────────

class TestAfterTTLExpiry:

    @pytest.mark.asyncio
    async def test_after_key_deleted_next_event_is_new_incident(self, redis, loader):
        """
        We can't wait 45 real seconds for TTL expiry in a unit test.
        Instead: manually delete the key (simulating expiry) and
        verify the next event is treated as a new incident.
        This is the correct way to test post-expiry behavior without
        making tests slow.
        """
        event = make_alert_event(confidence=0.85)
        first_result = await run_dedup(event, redis, loader)
        assert first_result.is_duplicate is False

        # Manually simulate TTL expiry
        await redis.delete(first_result.dedup_key)

        # Next detection — should be a fresh new incident
        second_result = await run_dedup(event, redis, loader)
        assert second_result.is_duplicate is False
        assert second_result.is_escalation is False


# ─────────────────────────────────────────────────────────────
# clear_dedup_key utility
# ─────────────────────────────────────────────────────────────

class TestClearDedupKey:

    @pytest.mark.asyncio
    async def test_clear_existing_key_returns_true(self, redis, loader):
        event = make_alert_event()
        await run_dedup(event, redis, loader)  # sets the key

        deleted = await clear_dedup_key(event, redis)
        assert deleted is True

    @pytest.mark.asyncio
    async def test_clear_nonexistent_key_returns_false(self, redis, loader):
        event = make_alert_event()
        # Never ran dedup — key doesn't exist
        deleted = await clear_dedup_key(event, redis)
        assert deleted is False

    @pytest.mark.asyncio
    async def test_after_clear_next_event_is_new_incident(self, redis, loader):
        """
        This is the operator-confirms-alert scenario: operator says
        "yes this is real, keep alerting." We clear the key so the
        next detection fires a fresh alert.
        """
        event = make_alert_event(confidence=0.85)
        await run_dedup(event, redis, loader)

        await clear_dedup_key(event, redis)

        # Next event should be treated as new
        next_event = make_alert_event(confidence=0.87)
        result = await run_dedup(next_event, redis, loader)
        assert result.is_duplicate is False