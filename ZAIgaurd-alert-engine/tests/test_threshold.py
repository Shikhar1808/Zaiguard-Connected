"""
Tests for config/thresholds.py
=================================
These tests manually populate ThresholdConfigCache instead of hitting
a real Postgres instance — they validate the CACHE READ LOGIC and the
effective_threshold FORMULA in isolation. This is deliberate: Layer 1
will depend on this logic being correct regardless of how the data
got into the cache, so we test that logic with zero external services.

A separate integration test (added once Step 9's test infra is up)
will verify _reload() actually pulls correct rows from a real
Postgres instance — that's a different concern (DB query correctness)
from this file's concern (cache read/formula correctness).

Run with:
    pytest tests/test_thresholds.py -v
"""

import pytest

from config.thresholds import (
    ConfigNotLoadedError,
    ThresholdConfigCache,
    ThresholdConfigLoader,
)


def make_populated_loader() -> ThresholdConfigLoader:
    """
    Builds a loader and manually injects a cache matching the
    real seed.sql data — so these tests double as a regression
    check that the formula produces the values documented in
    the architecture doc.
    """
    loader = ThresholdConfigLoader()
    cache = ThresholdConfigCache()

    cache.thresholds = {
        "fire":        (0.60, 120, 0.15),
        "violence":    (0.72, 45, 0.15),
        "dog_attack":  (0.68, 45, 0.15),
        "trespassing": (0.78, 60, 0.15),
        "accident":    (0.70, 60, 0.15),
    }

    cache.time_multipliers = [
        (22, 23, 0.85),
        (0, 5, 0.85),
        (6, 7, 1.00),
        (8, 18, 1.10),
        (19, 21, 1.00),
    ]

    cache.zone_multipliers = {
        "default": 1.00,
        "restricted_high": 0.85,
        "public_high": 1.15,
        "public_low": 1.00,
        "parking": 0.95,
        "entrance": 0.90,
    }

    cache.tier_rules = {
        "fire": [(0.60, "CRITICAL")],
        "violence": [(0.90, "CRITICAL"), (0.72, "HIGH")],
        "dog_attack": [(0.80, "HIGH"), (0.68, "MEDIUM")],
        "trespassing": [(0.78, "MEDIUM")],
        "accident": [(0.80, "HIGH"), (0.70, "MEDIUM")],
    }

    cache.similarity_thresholds = {
        "fire": 0.95,
        "violence": 0.88,
        "dog_attack": 0.85,
        "trespassing": 0.90,
        "accident": 0.93,
    }

    import time
    cache.loaded_at = time.monotonic()
    loader._cache = cache
    return loader


# ─────────────────────────────────────────────────────────────
# ConfigNotLoadedError — guard against using config before load
# ─────────────────────────────────────────────────────────────

class TestUnloadedConfig:

    def test_accessing_unloaded_config_raises(self):
        loader = ThresholdConfigLoader()
        with pytest.raises(ConfigNotLoadedError):
            loader.get_base_threshold("fire")

    def test_is_stale_true_before_first_load(self):
        loader = ThresholdConfigLoader()
        assert loader.is_stale is True


# ─────────────────────────────────────────────────────────────
# Time multiplier lookups
# ─────────────────────────────────────────────────────────────

class TestTimeMultiplier:

    def test_nighttime_late_hour(self):
        loader = make_populated_loader()
        assert loader.get_time_multiplier(22) == 0.85
        assert loader.get_time_multiplier(23) == 0.85

    def test_nighttime_early_hour(self):
        loader = make_populated_loader()
        assert loader.get_time_multiplier(0) == 0.85
        assert loader.get_time_multiplier(5) == 0.85

    def test_peak_hours(self):
        loader = make_populated_loader()
        assert loader.get_time_multiplier(8) == 1.10
        assert loader.get_time_multiplier(18) == 1.10

    def test_off_peak(self):
        loader = make_populated_loader()
        assert loader.get_time_multiplier(6) == 1.00
        assert loader.get_time_multiplier(19) == 1.00

    def test_unconfigured_hour_defaults_to_neutral(self):
        """A gap in config should never crash — default to 1.0."""
        loader = ThresholdConfigLoader()
        loader._cache.time_multipliers = []  # empty config
        import time
        loader._cache.loaded_at = time.monotonic()
        assert loader.get_time_multiplier(12) == 1.0


# ─────────────────────────────────────────────────────────────
# Zone multiplier lookups
# ─────────────────────────────────────────────────────────────

class TestZoneMultiplier:

    def test_known_zone(self):
        loader = make_populated_loader()
        assert loader.get_zone_multiplier("restricted_high") == 0.85
        assert loader.get_zone_multiplier("public_high") == 1.15

    def test_unknown_zone_defaults_to_neutral(self):
        """An unconfigured zone should not skew sensitivity."""
        loader = make_populated_loader()
        assert loader.get_zone_multiplier("some_new_zone_nobody_configured") == 1.0


# ─────────────────────────────────────────────────────────────
# Effective threshold formula — the core of Layer 1
# ─────────────────────────────────────────────────────────────

class TestEffectiveThreshold:

    def test_violence_at_peak_hours_in_high_traffic_zone(self):
        """
        0.72 (base) x 1.10 (peak) x 1.15 (public_high)
        = 0.9108
        Higher threshold = system becomes LESS sensitive during
        busy daytime hours in a noisy public zone — exactly the
        intended behavior to reduce false-positive fatigue.
        """
        loader = make_populated_loader()
        result = loader.get_effective_threshold("violence", hour=10, zone_id="public_high")
        assert result == pytest.approx(0.72 * 1.10 * 1.15)

    def test_fire_at_night_in_restricted_zone(self):
        """
        0.60 (base) x 0.85 (nighttime) x 0.85 (restricted_high)
        = 0.4335
        Lower threshold = MORE sensitive — exactly right for the
        highest-stakes event type at the highest-risk time/place.
        """
        loader = make_populated_loader()
        result = loader.get_effective_threshold("fire", hour=2, zone_id="restricted_high")
        assert result == pytest.approx(0.60 * 0.85 * 0.85)

    def test_unknown_pipeline_raises_keyerror(self):
        loader = make_populated_loader()
        with pytest.raises(KeyError):
            loader.get_effective_threshold("explosions", hour=12, zone_id="default")

    def test_default_zone_is_neutral(self):
        """default zone has risk_multiplier 1.0 — should equal base x time only."""
        loader = make_populated_loader()
        result = loader.get_effective_threshold("trespassing", hour=12, zone_id="default")
        expected_time_mult = loader.get_time_multiplier(12)
        assert result == pytest.approx(0.78 * expected_time_mult * 1.0)


# ─────────────────────────────────────────────────────────────
# Tier assignment
# ─────────────────────────────────────────────────────────────

class TestTierAssignment:

    def test_fire_always_critical(self):
        loader = make_populated_loader()
        assert loader.get_tier("fire", 0.61) == "CRITICAL"
        assert loader.get_tier("fire", 0.99) == "CRITICAL"

    def test_violence_high_vs_critical_boundary(self):
        loader = make_populated_loader()
        assert loader.get_tier("violence", 0.89) == "HIGH"
        assert loader.get_tier("violence", 0.90) == "CRITICAL"
        assert loader.get_tier("violence", 0.95) == "CRITICAL"

    def test_violence_just_above_base_threshold(self):
        loader = make_populated_loader()
        assert loader.get_tier("violence", 0.72) == "HIGH"

    def test_dog_attack_medium_vs_high(self):
        loader = make_populated_loader()
        assert loader.get_tier("dog_attack", 0.68) == "MEDIUM"
        assert loader.get_tier("dog_attack", 0.79) == "MEDIUM"
        assert loader.get_tier("dog_attack", 0.80) == "HIGH"

    def test_trespassing_always_medium(self):
        loader = make_populated_loader()
        assert loader.get_tier("trespassing", 0.78) == "MEDIUM"
        assert loader.get_tier("trespassing", 0.99) == "MEDIUM"

    def test_confidence_below_all_rules_falls_back_to_low(self):
        """
        An event that passed Layer 1 (so it's >= base_threshold) but
        somehow doesn't match any tier rule should still get displayed,
        not dropped. LOW is the safety-net tier.
        """
        loader = make_populated_loader()
        # accident's lowest tier rule starts at 0.70 — test just under
        assert loader.get_tier("accident", 0.65) == "LOW"

    def test_unknown_pipeline_falls_back_to_low(self):
        """No tier rules at all for this pipeline -> LOW, not a crash."""
        loader = make_populated_loader()
        assert loader.get_tier("unknown_pipeline", 0.99) == "LOW"


# ─────────────────────────────────────────────────────────────
# Similarity threshold lookups (used by Layer 4B later)
# ─────────────────────────────────────────────────────────────

class TestSimilarityThreshold:

    def test_known_pipeline(self):
        loader = make_populated_loader()
        assert loader.get_similarity_threshold("fire") == 0.95
        assert loader.get_similarity_threshold("violence") == 0.88

    def test_unknown_pipeline_defaults_conservative(self):
        """
        Missing config should fail toward SHOWING alerts (high
        similarity threshold = suppress almost nothing), never
        toward silently hiding genuine events.
        """
        loader = make_populated_loader()
        assert loader.get_similarity_threshold("unknown_pipeline") == 0.95


# ─────────────────────────────────────────────────────────────
# Staleness / TTL behavior
# ─────────────────────────────────────────────────────────────

class TestStaleness:

    def test_freshly_loaded_cache_is_not_stale(self):
        loader = make_populated_loader()
        assert loader.is_stale is False

    def test_custom_ttl_respected(self):
        import time
        loader = ThresholdConfigLoader(ttl_seconds=0)
        loader._cache = ThresholdConfigCache(loaded_at=time.monotonic())
        # TTL of 0 means it's immediately stale on next check
        time.sleep(0.01)
        assert loader.is_stale is True