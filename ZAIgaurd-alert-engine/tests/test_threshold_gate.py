"""
Tests for layers/threshold_gate.py
=====================================
All tests inject a pre-populated ThresholdConfigLoader so no live
Postgres is needed. The loader is identical to the one used in
test_thresholds.py — consistent seed data across the test suite.

Run with:
    pytest tests/test_threshold_gate.py -v
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest

from config.thresholds import ThresholdConfigCache, ThresholdConfigLoader
from layers.threshold_gate import ThresholdResult, run_threshold_gate
from models.schemas import Pipeline, RawDetectionEvent


# ─────────────────────────────────────────────────────────────
# Shared test loader (mirrors seed.sql values exactly)
# ─────────────────────────────────────────────────────────────

def make_loader() -> ThresholdConfigLoader:
    loader = ThresholdConfigLoader()
    cache = ThresholdConfigCache()
    cache.thresholds = {
        "fire":        (0.60, 120, 0.15),
        "violence":    (0.72,  45, 0.15),
        "dog_attack":  (0.68,  45, 0.15),
        "trespassing": (0.78,  60, 0.15),
        "accident":    (0.70,  60, 0.15),
    }
    cache.time_multipliers = [
        (22, 23, 0.85),
        ( 0,  5, 0.85),
        ( 6,  7, 1.00),
        ( 8, 18, 1.10),
        (19, 21, 1.00),
    ]
    cache.zone_multipliers = {
        "default":         1.00,
        "restricted_high": 0.85,
        "public_high":     1.15,
        "parking":         0.95,
        "entrance":        0.90,
    }
    cache.tier_rules = {}
    cache.similarity_thresholds = {}
    cache.loaded_at = time.monotonic()
    loader._cache = cache
    return loader


def make_event(
    pipeline: Pipeline = Pipeline.VIOLENCE,
    confidence: float = 0.80,
    hour: int = 12,
    zone_id: str = "default",
    camera_id: str = "CAM_01",
) -> RawDetectionEvent:
    return RawDetectionEvent(
        pipeline=pipeline,
        raw_confidence=confidence,
        camera_id=camera_id,
        zone_id=zone_id,
        timestamp=datetime(2026, 6, 8, hour, 0, 0, tzinfo=timezone.utc),
    )


# ─────────────────────────────────────────────────────────────
# ThresholdResult shape
# ─────────────────────────────────────────────────────────────

class TestThresholdResultShape:

    @pytest.mark.asyncio
    async def test_result_fields_populated(self):
        event = make_event(confidence=0.80, hour=12, zone_id="default")
        result = await run_threshold_gate(event, loader=make_loader())

        assert isinstance(result, ThresholdResult)
        assert result.raw_confidence == 0.80
        assert result.effective_conf == 0.80
        assert result.pipeline == Pipeline.VIOLENCE
        assert result.camera_id == "CAM_01"
        assert result.zone_id == "default"
        assert result.hour_of_day == 12

    @pytest.mark.asyncio
    async def test_result_is_immutable(self):
        """frozen=True on the dataclass — should not be modifiable."""
        event = make_event()
        result = await run_threshold_gate(event, loader=make_loader())
        with pytest.raises(Exception):
            result.passed = not result.passed  # type: ignore


# ─────────────────────────────────────────────────────────────
# Basic pass / fail
# ─────────────────────────────────────────────────────────────

class TestPassFail:

    @pytest.mark.asyncio
    async def test_high_confidence_passes(self):
        """0.90 well above violence base threshold of 0.72."""
        event = make_event(pipeline=Pipeline.VIOLENCE, confidence=0.90, hour=12)
        result = await run_threshold_gate(event, loader=make_loader())
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_low_confidence_fails(self):
        """0.50 well below any pipeline's threshold."""
        event = make_event(pipeline=Pipeline.VIOLENCE, confidence=0.50, hour=12)
        result = await run_threshold_gate(event, loader=make_loader())
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_exact_boundary_passes(self):
        """
        Boundary condition: confidence == effective_threshold should pass.
        The check is >= not >, so the boundary value is inclusive.
        This test explicitly pins that contract.
        """
        loader = make_loader()
        # violence base 0.72, peak hours x1.10, default zone x1.0
        # effective = 0.72 * 1.10 * 1.0 = 0.792
        effective = 0.72 * 1.10 * 1.0
        event = make_event(
            pipeline=Pipeline.VIOLENCE,
            confidence=effective,
            hour=12,       # peak hours
            zone_id="default",
        )
        result = await run_threshold_gate(event, loader=loader)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_one_below_boundary_fails(self):
        """0.001 below the effective threshold should fail."""
        loader = make_loader()
        effective = 0.72 * 1.10 * 1.0
        event = make_event(
            pipeline=Pipeline.VIOLENCE,
            confidence=round(effective - 0.001, 4),
            hour=12,
            zone_id="default",
        )
        result = await run_threshold_gate(event, loader=loader)
        assert result.passed is False


# ─────────────────────────────────────────────────────────────
# All five pipelines — base thresholds at neutral time/zone
# ─────────────────────────────────────────────────────────────

class TestAllPipelines:
    """
    Verifies each pipeline's base threshold at hour=7 (off-peak, x1.0)
    and zone=default (x1.0) — so effective_threshold == base_threshold.
    Tests both just-above-threshold (pass) and just-below (fail).
    """

    PIPELINE_THRESHOLDS = [
        (Pipeline.FIRE,        0.60),
        (Pipeline.VIOLENCE,    0.72),
        (Pipeline.DOG_ATTACK,  0.68),
        (Pipeline.TRESPASSING, 0.78),
        (Pipeline.ACCIDENT,    0.70),
    ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("pipeline,base", PIPELINE_THRESHOLDS)
    async def test_just_above_base_threshold_passes(self, pipeline, base):
        event = make_event(pipeline=pipeline, confidence=base + 0.01, hour=7)
        result = await run_threshold_gate(event, loader=make_loader())
        assert result.passed is True, (
            f"{pipeline.value}: {base + 0.01} should pass base threshold {base}"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("pipeline,base", PIPELINE_THRESHOLDS)
    async def test_just_below_base_threshold_fails(self, pipeline, base):
        event = make_event(pipeline=pipeline, confidence=base - 0.01, hour=7)
        result = await run_threshold_gate(event, loader=make_loader())
        assert result.passed is False, (
            f"{pipeline.value}: {base - 0.01} should fail base threshold {base}"
        )


# ─────────────────────────────────────────────────────────────
# Time multiplier effects
# ─────────────────────────────────────────────────────────────

class TestTimeMultiplierEffect:

    @pytest.mark.asyncio
    async def test_nighttime_lowers_effective_threshold(self):
        """
        At night (hour=2), the multiplier is 0.85, so effective_threshold
        is lower than daytime. An event that would FAIL at peak hours
        should PASS at night for the same confidence level.

        Violence base = 0.72
        Peak (hour=12):   0.72 * 1.10 = 0.792  ← threshold is higher
        Nighttime (hour=2): 0.72 * 0.85 = 0.612 ← threshold is lower
        """
        loader = make_loader()
        # 0.75 fails peak (threshold 0.792) but passes nighttime (threshold 0.612)
        event_peak = make_event(pipeline=Pipeline.VIOLENCE, confidence=0.75, hour=12)
        event_night = make_event(pipeline=Pipeline.VIOLENCE, confidence=0.75, hour=2)

        result_peak = await run_threshold_gate(event_peak, loader=loader)
        result_night = await run_threshold_gate(event_night, loader=loader)

        assert result_peak.passed is False
        assert result_night.passed is True

    @pytest.mark.asyncio
    async def test_peak_hours_raises_effective_threshold(self):
        """
        At peak hours (hour=12), the multiplier is 1.10, raising the
        threshold — the system demands higher confidence during busy
        periods to reduce false-positive noise.
        """
        loader = make_loader()
        peak_threshold = 0.72 * 1.10
        # Confidence just below the raised threshold should fail
        event = make_event(
            pipeline=Pipeline.VIOLENCE,
            confidence=peak_threshold - 0.01,
            hour=12,
        )
        result = await run_threshold_gate(event, loader=loader)
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_effective_threshold_stored_in_result(self):
        """
        The result should carry the computed effective_threshold so
        callers can log or display it without recomputing.
        """
        loader = make_loader()
        event = make_event(pipeline=Pipeline.FIRE, confidence=0.70, hour=2)
        result = await run_threshold_gate(event, loader=loader)
        expected = 0.60 * 0.85 * 1.0   # fire * nighttime * default_zone
        assert result.effective_threshold == pytest.approx(expected)


# ─────────────────────────────────────────────────────────────
# Zone multiplier effects
# ─────────────────────────────────────────────────────────────

class TestZoneMultiplierEffect:

    @pytest.mark.asyncio
    async def test_restricted_zone_lowers_threshold(self):
        """
        restricted_high has multiplier 0.85 — more sensitive.
        An event that fails in the default zone passes in the
        restricted zone at the same confidence and time.

        Violence base 0.72 * peak 1.10 * default 1.0  = 0.792
        Violence base 0.72 * peak 1.10 * restricted 0.85 = 0.6732
        """
        loader = make_loader()
        confidence = 0.75  # fails default at peak, passes restricted at peak

        result_default = await run_threshold_gate(
            make_event(confidence=confidence, hour=12, zone_id="default"),
            loader=loader,
        )
        result_restricted = await run_threshold_gate(
            make_event(confidence=confidence, hour=12, zone_id="restricted_high"),
            loader=loader,
        )

        assert result_default.passed is False
        assert result_restricted.passed is True

    @pytest.mark.asyncio
    async def test_high_traffic_zone_raises_threshold(self):
        """
        public_high has multiplier 1.15 — less sensitive (noisy area).
        """
        loader = make_loader()
        # violence * off-peak * public_high = 0.72 * 1.0 * 1.15 = 0.828
        threshold = 0.72 * 1.0 * 1.15
        event = make_event(
            confidence=threshold - 0.01,
            hour=7,            # off-peak
            zone_id="public_high",
        )
        result = await run_threshold_gate(event, loader=loader)
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_unknown_zone_uses_neutral_multiplier(self):
        """
        An unconfigured zone_id defaults to 1.0 multiplier —
        no effect on the threshold. Should behave identically
        to the 'default' zone.
        """
        loader = make_loader()
        event_default = make_event(confidence=0.80, hour=7, zone_id="default")
        event_unknown = make_event(confidence=0.80, hour=7, zone_id="never_configured_zone")

        result_default = await run_threshold_gate(event_default, loader=loader)
        result_unknown = await run_threshold_gate(event_unknown, loader=loader)

        assert result_default.passed == result_unknown.passed
        assert result_default.effective_threshold == pytest.approx(
            result_unknown.effective_threshold
        )


# ─────────────────────────────────────────────────────────────
# Combined multiplier scenarios
# ─────────────────────────────────────────────────────────────

class TestCombinedMultipliers:

    @pytest.mark.asyncio
    async def test_most_sensitive_combination(self):
        """
        Nighttime (0.85) + restricted_high (0.85) = lowest possible threshold.
        Even fire's already-low base threshold (0.60) gets reduced further:
        0.60 * 0.85 * 0.85 = 0.4335
        A barely-there confidence of 0.45 should pass.
        """
        loader = make_loader()
        event = make_event(
            pipeline=Pipeline.FIRE,
            confidence=0.45,
            hour=2,
            zone_id="restricted_high",
        )
        result = await run_threshold_gate(event, loader=loader)
        assert result.passed is True
        assert result.effective_threshold == pytest.approx(0.60 * 0.85 * 0.85)

    @pytest.mark.asyncio
    async def test_least_sensitive_combination(self):
        """
        Peak hours (1.10) + public_high (1.15) = highest possible threshold.
        Trespassing base (0.78) * 1.10 * 1.15 = 0.9867
        A confidence of 0.95 — high for any model — should still fail.
        """
        loader = make_loader()
        event = make_event(
            pipeline=Pipeline.TRESPASSING,
            confidence=0.95,
            hour=12,
            zone_id="public_high",
        )
        result = await run_threshold_gate(event, loader=loader)
        assert result.passed is False
        assert result.effective_threshold == pytest.approx(0.78 * 1.10 * 1.15)