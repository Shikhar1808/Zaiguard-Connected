"""
Tests for layers/enrichment.py
=================================
Enrichment is synchronous and has no external dependencies — these
tests run completely in isolation with zero services needed.

Run with:
    pytest tests/test_enrichment.py -v
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest

from config.thresholds import ThresholdConfigCache, ThresholdConfigLoader
from layers.enrichment import run_enrichment
from layers.threshold_gate import ThresholdResult, run_threshold_gate
from models.schemas import AlertEvent, Pipeline, RawDetectionEvent, build_alert_id


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def make_loader() -> ThresholdConfigLoader:
    loader = ThresholdConfigLoader()
    cache = ThresholdConfigCache()
    cache.thresholds = {
        "violence": (0.72, 45, 0.15),
        "fire":     (0.60, 120, 0.15),
    }
    cache.time_multipliers = [(8, 18, 1.10), (0, 7, 1.00), (19, 23, 1.00)]
    cache.zone_multipliers = {"default": 1.0}
    cache.tier_rules = {}
    cache.similarity_thresholds = {}
    cache.loaded_at = time.monotonic()
    loader._cache = cache
    return loader


def make_event(
    pipeline: Pipeline = Pipeline.VIOLENCE,
    confidence: float = 0.85,
    hour: int = 12,
    day: int = 8,        # June 8 2026 = Monday (weekday 0)
    camera_id: str = "CAM_07",
    zone_id: str = "default",
    zone_label: str = "Default Zone",
    involved_ids: list[int] | None = None,
) -> RawDetectionEvent:
    return RawDetectionEvent(
        pipeline=pipeline,
        raw_confidence=confidence,
        camera_id=camera_id,
        zone_id=zone_id,
        zone_label=zone_label,
        timestamp=datetime(2026, 6, day, hour, 0, 0, tzinfo=timezone.utc),
        frame_ref=f"/clips/CAM_07_{hour:02d}-00-00.mp4",
        involved_ids=involved_ids or [3, 7],
        pipeline_features={"inter_person_distance": 0.4},
    )


def make_threshold_result(
    event: RawDetectionEvent,
    effective_threshold: float = 0.792,
) -> ThresholdResult:
    """Build a synthetic ThresholdResult that says the event passed."""
    return ThresholdResult(
        passed=True,
        effective_conf=event.raw_confidence,
        effective_threshold=effective_threshold,
        raw_confidence=event.raw_confidence,
        pipeline=event.pipeline,
        camera_id=event.camera_id,
        zone_id=event.zone_id,
        hour_of_day=event.timestamp.hour,
    )


# ─────────────────────────────────────────────────────────────
# AlertEvent construction
# ─────────────────────────────────────────────────────────────

class TestAlertEventConstruction:

    def test_returns_alert_event(self):
        event = make_event()
        result = make_threshold_result(event)
        enriched = run_enrichment(event, result)
        assert isinstance(enriched, AlertEvent)

    def test_source_event_preserved(self):
        """The original RawDetectionEvent should be accessible in full."""
        event = make_event()
        enriched = run_enrichment(event, make_threshold_result(event))
        assert enriched.source_event is event

    def test_effective_conf_taken_from_threshold_result(self):
        event = make_event(confidence=0.85)
        threshold_result = make_threshold_result(event, effective_threshold=0.792)
        enriched = run_enrichment(event, threshold_result)
        assert enriched.effective_conf == 0.85

    def test_pipeline_features_accessible(self):
        """pipeline_features from the raw event must survive enrichment."""
        event = make_event()
        enriched = run_enrichment(event, make_threshold_result(event))
        assert enriched.source_event.pipeline_features == {"inter_person_distance": 0.4}

    def test_frame_ref_preserved(self):
        event = make_event(hour=17)
        enriched = run_enrichment(event, make_threshold_result(event))
        assert enriched.frame_ref == "/clips/CAM_07_17-00-00.mp4"

    def test_involved_ids_preserved(self):
        event = make_event(involved_ids=[1, 2, 3])
        enriched = run_enrichment(event, make_threshold_result(event))
        assert enriched.involved_ids == [1, 2, 3]


# ─────────────────────────────────────────────────────────────
# Convenience property delegation
# ─────────────────────────────────────────────────────────────

class TestConvenienceProperties:
    """
    AlertEvent exposes properties that delegate to source_event.
    Verify they return the right values rather than just existing.
    """

    def test_pipeline_property(self):
        event = make_event(pipeline=Pipeline.FIRE)
        enriched = run_enrichment(event, make_threshold_result(event))
        assert enriched.pipeline == Pipeline.FIRE

    def test_camera_id_property(self):
        event = make_event(camera_id="CAM_42")
        enriched = run_enrichment(event, make_threshold_result(event))
        assert enriched.camera_id == "CAM_42"

    def test_zone_id_property(self):
        event = make_event(zone_id="parking")
        enriched = run_enrichment(event, make_threshold_result(event))
        assert enriched.zone_id == "parking"

    def test_raw_confidence_property(self):
        event = make_event(confidence=0.91)
        enriched = run_enrichment(event, make_threshold_result(event))
        assert enriched.raw_confidence == 0.91


# ─────────────────────────────────────────────────────────────
# Time field extraction
# ─────────────────────────────────────────────────────────────

class TestTimeFieldExtraction:

    def test_hour_of_day_extracted_correctly(self):
        for hour in [0, 6, 12, 17, 23]:
            event = make_event(hour=hour)
            enriched = run_enrichment(event, make_threshold_result(event))
            assert enriched.hour_of_day == hour, f"Failed for hour={hour}"

    def test_day_of_week_monday(self):
        # 2026-06-08 is a Monday → weekday() == 0
        event = make_event(day=8)
        enriched = run_enrichment(event, make_threshold_result(event))
        assert enriched.day_of_week == 0

    def test_day_of_week_sunday(self):
        # 2026-06-14 is a Sunday → weekday() == 6
        event = make_event(day=14)
        enriched = run_enrichment(event, make_threshold_result(event))
        assert enriched.day_of_week == 6

    def test_day_of_week_saturday(self):
        # 2026-06-13 is a Saturday → weekday() == 5
        event = make_event(day=13)
        enriched = run_enrichment(event, make_threshold_result(event))
        assert enriched.day_of_week == 5


# ─────────────────────────────────────────────────────────────
# alert_id generation — idempotency guarantees
# ─────────────────────────────────────────────────────────────

class TestAlertId:

    def test_same_event_produces_same_alert_id(self):
        """
        Core idempotency guarantee: if the same detection arrives
        twice (e.g. message bus retry), both produce the same
        alert_id. The DB unique constraint then rejects the second.
        """
        event = make_event()
        enriched_1 = run_enrichment(event, make_threshold_result(event))
        enriched_2 = run_enrichment(event, make_threshold_result(event))
        assert enriched_1.alert_id == enriched_2.alert_id

    def test_different_camera_produces_different_alert_id(self):
        event_a = make_event(camera_id="CAM_01")
        event_b = make_event(camera_id="CAM_02")
        enriched_a = run_enrichment(event_a, make_threshold_result(event_a))
        enriched_b = run_enrichment(event_b, make_threshold_result(event_b))
        assert enriched_a.alert_id != enriched_b.alert_id

    def test_different_timestamp_produces_different_alert_id(self):
        event_a = make_event(hour=10)
        event_b = make_event(hour=11)
        enriched_a = run_enrichment(event_a, make_threshold_result(event_a))
        enriched_b = run_enrichment(event_b, make_threshold_result(event_b))
        assert enriched_a.alert_id != enriched_b.alert_id

    def test_different_pipeline_produces_different_alert_id(self):
        event_a = make_event(pipeline=Pipeline.VIOLENCE)
        event_b = make_event(pipeline=Pipeline.FIRE)
        # Need same timestamp/camera for a fair comparison
        enriched_a = run_enrichment(event_a, make_threshold_result(event_a))
        enriched_b = run_enrichment(event_b, make_threshold_result(event_b))
        assert enriched_a.alert_id != enriched_b.alert_id

    def test_alert_id_is_32_hex_chars(self):
        event = make_event()
        enriched = run_enrichment(event, make_threshold_result(event))
        assert len(enriched.alert_id) == 32
        assert all(c in "0123456789abcdef" for c in enriched.alert_id)

    def test_alert_id_matches_build_alert_id_directly(self):
        """
        Enrichment must use build_alert_id() with the same inputs.
        This test catches any accidental divergence.
        """
        event = make_event()
        enriched = run_enrichment(event, make_threshold_result(event))
        expected = build_alert_id(event.camera_id, event.timestamp, event.pipeline)
        assert enriched.alert_id == expected


# ─────────────────────────────────────────────────────────────
# Layers 1 + 2 integrated — end-to-end through both gates
# ─────────────────────────────────────────────────────────────

class TestLayers1And2Integrated:
    """
    Runs a RawDetectionEvent through Layer 1 then Layer 2 together,
    verifying the two layers hand off correctly.
    """

    @pytest.mark.asyncio
    async def test_passing_event_produces_alert_event(self):
        loader = make_loader()
        event = make_event(confidence=0.90, hour=7)   # off-peak, confidence well above base

        gate_result = await run_threshold_gate(event, loader=loader)
        assert gate_result.passed is True

        enriched = run_enrichment(event, gate_result)
        assert enriched.alert_id is not None
        assert enriched.effective_conf == 0.90
        assert enriched.hour_of_day == 7
        assert enriched.camera_id == "CAM_07"

    @pytest.mark.asyncio
    async def test_failing_event_not_enriched(self):
        """
        Enrichment is only called on passing events. This test
        documents the contract: if Layer 1 says no, Layer 2 is
        never called. (pipeline.py enforces this — tested in Step 9.)
        """
        loader = make_loader()
        event = make_event(confidence=0.30, hour=12)  # well below threshold

        gate_result = await run_threshold_gate(event, loader=loader)
        assert gate_result.passed is False
        # If pipeline.py correctly checks gate_result.passed, enrichment
        # never runs for this event — verified in test_pipeline.py Step 9