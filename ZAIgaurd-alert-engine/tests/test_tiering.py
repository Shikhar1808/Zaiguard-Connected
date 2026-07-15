"""
Tests for layers/tiering.py (Layer 5)
======================================
Pure-logic tests — no external services needed.
All tests use a manually populated ThresholdConfigLoader cache
matching the values from seed.sql.

Run with:
    pytest tests/test_tiering.py -v
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone

import pytest

from config.thresholds import ThresholdConfigCache, ThresholdConfigLoader
from layers.tiering import TierResult, build_dashboard_alert, run_tiering
from models.schemas import (
    AlertEvent,
    AlertTier,
    DashboardAlert,
    Pipeline,
    RawDetectionEvent,
)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def make_loader() -> ThresholdConfigLoader:
    """Loader pre-populated with seed.sql values."""
    loader = ThresholdConfigLoader()
    cache = ThresholdConfigCache()
    cache.thresholds = {
        "fire":        (0.60, 120, 0.15),
        "violence":    (0.72,  45, 0.15),
        "dog_attack":  (0.68,  45, 0.15),
        "trespassing": (0.78,  60, 0.15),
        "accident":    (0.70,  60, 0.15),
    }
    cache.time_multipliers = []
    cache.zone_multipliers = {}
    cache.tier_rules = {
        "fire":        [(0.60, "CRITICAL")],
        "violence":    [(0.90, "CRITICAL"), (0.72, "HIGH")],
        "dog_attack":  [(0.80, "HIGH"),     (0.68, "MEDIUM")],
        "trespassing": [(0.78, "MEDIUM")],
        "accident":    [(0.80, "HIGH"),     (0.70, "MEDIUM")],
    }
    cache.similarity_thresholds = {}
    cache.loaded_at = time.monotonic()
    loader._cache = cache
    return loader


def make_event(
    pipeline: Pipeline = Pipeline.VIOLENCE,
    confidence: float = 0.85,
    camera_id: str = "CAM_07",
    zone_id: str = "gym_east",
) -> AlertEvent:
    raw = RawDetectionEvent(
        pipeline=pipeline,
        raw_confidence=confidence,
        camera_id=camera_id,
        zone_id=zone_id,
        zone_label="Gym East",
        timestamp=datetime(2026, 6, 8, 17, 0, 0, tzinfo=timezone.utc),
        frame_ref="/clips/test.mp4",
        involved_ids=[1, 2],
    )
    return AlertEvent(
        alert_id=f"test_{uuid.uuid4().hex[:8]}",
        source_event=raw,
        effective_conf=confidence,
        hour_of_day=17,
        day_of_week=0,
    )


# ─────────────────────────────────────────────────────────────
# TierResult dataclass
# ─────────────────────────────────────────────────────────────

class TestTierResult:

    def test_is_frozen(self):
        result = TierResult(tier=AlertTier.HIGH, pipeline="violence")
        with pytest.raises((AttributeError, TypeError)):
            result.tier = AlertTier.LOW  # type: ignore

    def test_fields_accessible(self):
        result = TierResult(tier=AlertTier.CRITICAL, pipeline="fire")
        assert result.tier == AlertTier.CRITICAL
        assert result.pipeline == "fire"


# ─────────────────────────────────────────────────────────────
# Fire — always CRITICAL above base threshold
# ─────────────────────────────────────────────────────────────

class TestFireTiering:

    @pytest.mark.asyncio
    async def test_fire_at_base_threshold_is_critical(self):
        loader = make_loader()
        event = make_event(pipeline=Pipeline.FIRE, confidence=0.60)
        result = await run_tiering(event, loader)
        assert result.tier == AlertTier.CRITICAL

    @pytest.mark.asyncio
    async def test_fire_at_high_confidence_is_critical(self):
        loader = make_loader()
        event = make_event(pipeline=Pipeline.FIRE, confidence=0.99)
        result = await run_tiering(event, loader)
        assert result.tier == AlertTier.CRITICAL

    @pytest.mark.asyncio
    async def test_fire_pipeline_echoed_in_result(self):
        loader = make_loader()
        event = make_event(pipeline=Pipeline.FIRE, confidence=0.80)
        result = await run_tiering(event, loader)
        assert result.pipeline == "fire"


# ─────────────────────────────────────────────────────────────
# Violence — CRITICAL ≥0.90, HIGH ≥0.72
# ─────────────────────────────────────────────────────────────

class TestViolenceTiering:

    @pytest.mark.asyncio
    async def test_violence_at_base_threshold_is_high(self):
        loader = make_loader()
        event = make_event(pipeline=Pipeline.VIOLENCE, confidence=0.72)
        result = await run_tiering(event, loader)
        assert result.tier == AlertTier.HIGH

    @pytest.mark.asyncio
    async def test_violence_below_critical_boundary_is_high(self):
        loader = make_loader()
        event = make_event(pipeline=Pipeline.VIOLENCE, confidence=0.899)
        result = await run_tiering(event, loader)
        assert result.tier == AlertTier.HIGH

    @pytest.mark.asyncio
    async def test_violence_at_critical_boundary_is_critical(self):
        """Boundary is inclusive — 0.90 exactly must be CRITICAL."""
        loader = make_loader()
        event = make_event(pipeline=Pipeline.VIOLENCE, confidence=0.90)
        result = await run_tiering(event, loader)
        assert result.tier == AlertTier.CRITICAL

    @pytest.mark.asyncio
    async def test_violence_above_critical_boundary_is_critical(self):
        loader = make_loader()
        event = make_event(pipeline=Pipeline.VIOLENCE, confidence=0.95)
        result = await run_tiering(event, loader)
        assert result.tier == AlertTier.CRITICAL


# ─────────────────────────────────────────────────────────────
# Dog attack — HIGH ≥0.80, MEDIUM ≥0.68
# ─────────────────────────────────────────────────────────────

class TestDogAttackTiering:

    @pytest.mark.asyncio
    async def test_dog_attack_at_base_threshold_is_medium(self):
        loader = make_loader()
        event = make_event(pipeline=Pipeline.DOG_ATTACK, confidence=0.68)
        result = await run_tiering(event, loader)
        assert result.tier == AlertTier.MEDIUM

    @pytest.mark.asyncio
    async def test_dog_attack_just_below_high_is_medium(self):
        loader = make_loader()
        event = make_event(pipeline=Pipeline.DOG_ATTACK, confidence=0.799)
        result = await run_tiering(event, loader)
        assert result.tier == AlertTier.MEDIUM

    @pytest.mark.asyncio
    async def test_dog_attack_at_high_boundary_is_high(self):
        loader = make_loader()
        event = make_event(pipeline=Pipeline.DOG_ATTACK, confidence=0.80)
        result = await run_tiering(event, loader)
        assert result.tier == AlertTier.HIGH

    @pytest.mark.asyncio
    async def test_dog_attack_above_high_boundary_is_high(self):
        loader = make_loader()
        event = make_event(pipeline=Pipeline.DOG_ATTACK, confidence=0.95)
        result = await run_tiering(event, loader)
        assert result.tier == AlertTier.HIGH


# ─────────────────────────────────────────────────────────────
# Trespassing — always MEDIUM above base threshold
# ─────────────────────────────────────────────────────────────

class TestTrespassingTiering:

    @pytest.mark.asyncio
    async def test_trespassing_always_medium(self):
        loader = make_loader()
        for conf in [0.78, 0.85, 0.95, 1.0]:
            event = make_event(pipeline=Pipeline.TRESPASSING, confidence=conf)
            result = await run_tiering(event, loader)
            assert result.tier == AlertTier.MEDIUM, (
                f"Expected MEDIUM at confidence {conf}, got {result.tier}"
            )


# ─────────────────────────────────────────────────────────────
# Accident — HIGH ≥0.80, MEDIUM ≥0.70
# ─────────────────────────────────────────────────────────────

class TestAccidentTiering:

    @pytest.mark.asyncio
    async def test_accident_at_base_threshold_is_medium(self):
        loader = make_loader()
        event = make_event(pipeline=Pipeline.ACCIDENT, confidence=0.70)
        result = await run_tiering(event, loader)
        assert result.tier == AlertTier.MEDIUM

    @pytest.mark.asyncio
    async def test_accident_at_high_boundary_is_high(self):
        loader = make_loader()
        event = make_event(pipeline=Pipeline.ACCIDENT, confidence=0.80)
        result = await run_tiering(event, loader)
        assert result.tier == AlertTier.HIGH


# ─────────────────────────────────────────────────────────────
# LOW fallback — confidence below all tier rules
# ─────────────────────────────────────────────────────────────

class TestLowFallback:

    @pytest.mark.asyncio
    async def test_confidence_below_all_rules_gets_low(self):
        """
        An event below all tier rule thresholds should get LOW,
        not crash. LOW is the safety-net tier — the event reached
        the dashboard because it passed Layer 1 at base threshold,
        but none of the tier rules apply at this confidence level.
        """
        loader = make_loader()
        event = make_event(pipeline=Pipeline.ACCIDENT, confidence=0.65)
        result = await run_tiering(event, loader)
        assert result.tier == AlertTier.LOW

    @pytest.mark.asyncio
    async def test_unknown_pipeline_gets_low_not_crash(self):
        """
        If a new pipeline is added to the classifiers but not yet
        to tier_config, it should gracefully fall back to LOW.
        """
        loader = make_loader()
        # Manually remove accident tier rules to simulate unconfigured pipeline
        loader._cache.tier_rules.pop("accident", None)
        event = make_event(pipeline=Pipeline.ACCIDENT, confidence=0.85)
        result = await run_tiering(event, loader)
        assert result.tier == AlertTier.LOW


# ─────────────────────────────────────────────────────────────
# build_dashboard_alert — final object assembly
# ─────────────────────────────────────────────────────────────

class TestBuildDashboardAlert:

    def test_builds_correct_dashboard_alert(self):
        event = make_event(pipeline=Pipeline.VIOLENCE, confidence=0.88)
        alert = build_dashboard_alert(event, AlertTier.HIGH, suppression_score=0.72)

        assert alert.alert_id == event.alert_id
        assert alert.tier == AlertTier.HIGH
        assert alert.pipeline == Pipeline.VIOLENCE
        assert alert.camera_id == "CAM_07"
        assert alert.zone_id == "gym_east"
        assert alert.zone_label == "Gym East"
        assert alert.raw_confidence == pytest.approx(0.88)
        assert alert.effective_conf == pytest.approx(0.88)
        assert alert.suppression_score == pytest.approx(0.72)
        assert alert.evidence_frame_ref == "/clips/test.mp4"
        assert alert.involved_ids == [1, 2]

    def test_suppression_score_defaults_to_zero(self):
        event = make_event()
        alert = build_dashboard_alert(event, AlertTier.MEDIUM)
        assert alert.suppression_score == 0.0

    def test_returns_dashboard_alert_instance(self):
        event = make_event()
        alert = build_dashboard_alert(event, AlertTier.LOW)
        assert isinstance(alert, DashboardAlert)

    def test_all_five_pipelines_build_correctly(self):
        for pipeline in Pipeline:
            event = make_event(pipeline=pipeline, confidence=0.80)
            alert = build_dashboard_alert(event, AlertTier.MEDIUM)
            assert alert.pipeline == pipeline

    def test_timestamp_preserved(self):
        event = make_event()
        alert = build_dashboard_alert(event, AlertTier.HIGH)
        assert alert.timestamp == event.timestamp

    def test_tier_is_alert_tier_enum(self):
        event = make_event()
        alert = build_dashboard_alert(event, AlertTier.CRITICAL)
        assert isinstance(alert.tier, AlertTier)
        assert alert.tier == AlertTier.CRITICAL


# ─────────────────────────────────────────────────────────────
# Integration: run_tiering → build_dashboard_alert
# ─────────────────────────────────────────────────────────────

class TestTieringToDashboardAlertIntegration:

    @pytest.mark.asyncio
    async def test_full_flow_violence_critical(self):
        """
        Simulates the final two steps in pipeline.py:
        run_tiering() → build_dashboard_alert()
        Verifies the complete handoff produces a correct DashboardAlert.
        """
        loader = make_loader()
        event = make_event(pipeline=Pipeline.VIOLENCE, confidence=0.93)

        tier_result = await run_tiering(event, loader)
        alert = build_dashboard_alert(event, tier_result.tier, suppression_score=0.41)

        assert alert.tier == AlertTier.CRITICAL
        assert alert.suppression_score == pytest.approx(0.41)
        assert alert.pipeline == Pipeline.VIOLENCE

    @pytest.mark.asyncio
    async def test_full_flow_fire_always_critical(self):
        loader = make_loader()
        event = make_event(pipeline=Pipeline.FIRE, confidence=0.62)

        tier_result = await run_tiering(event, loader)
        alert = build_dashboard_alert(event, tier_result.tier)

        assert alert.tier == AlertTier.CRITICAL

    @pytest.mark.asyncio
    async def test_full_flow_all_pipelines_produce_valid_alert(self):
        """Every pipeline produces a valid DashboardAlert — no crashes."""
        loader = make_loader()
        confidences = {
            Pipeline.FIRE:        0.75,
            Pipeline.VIOLENCE:    0.80,
            Pipeline.DOG_ATTACK:  0.72,
            Pipeline.TRESPASSING: 0.82,
            Pipeline.ACCIDENT:    0.75,
        }
        for pipeline, conf in confidences.items():
            event = make_event(pipeline=pipeline, confidence=conf)
            tier_result = await run_tiering(event, loader)
            alert = build_dashboard_alert(event, tier_result.tier)
            assert isinstance(alert, DashboardAlert)
            assert isinstance(alert.tier, AlertTier)