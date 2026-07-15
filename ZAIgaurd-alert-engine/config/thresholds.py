"""
ZaiGuard Alert Engine — Threshold & Tier Configuration Loader
=================================================================
Loads threshold_config, time_multiplier_config, zone_config,
tier_config, and suppression_similarity_config from Postgres into
an in-memory cache, refreshed periodically.

WHY CACHE IN MEMORY AT ALL
-----------------------------
Layer 1 (threshold gate) runs on every single incoming event —
potentially thousands per second across many cameras. Querying
Postgres on every event for config that changes maybe once a day
(an operator tweaking a threshold) would be wasteful and add
unnecessary latency to the hot path. Instead: load once, cache,
refresh on a timer. Operators see their changes take effect within
CONFIG_CACHE_TTL_SECONDS (default 60s) — fast enough to feel
responsive, far cheaper than a DB round-trip per event.

WHY NOT CACHE FOREVER
------------------------
The whole point of storing config in Postgres instead of hardcoding
it (see architecture doc §3) is that operators can tune it from the
dashboard without a code deploy. A cache that never refreshes would
defeat that purpose. The TTL is the balance point between "fast"
and "reflects operator changes promptly."
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from sqlalchemy import select

from config.database import db_session_scope
from config.settings import settings
from models.db_models import (
    SuppressionSimilarityConfigRow,
    ThresholdConfigRow,
    TierConfigRow,
    TimeMultiplierConfigRow,
    ZoneConfigRow,
)


# ─────────────────────────────────────────────────────────────
# Cache data structure
# ─────────────────────────────────────────────────────────────

@dataclass
class ThresholdConfigCache:
    """
    In-memory snapshot of every config table.
    Rebuilt wholesale on each refresh — simpler and safer than
    trying to patch individual rows, and these tables are small
    (a handful to a few dozen rows each), so a full reload is cheap.
    """

    # pipeline -> (base_threshold, dedup_ttl_seconds, escalation_delta)
    thresholds: dict[str, tuple[float, int, float]] = field(default_factory=dict)

    # list of (hour_start, hour_end, multiplier) — checked in order
    time_multipliers: list[tuple[int, int, float]] = field(default_factory=list)

    # zone_id -> risk_multiplier
    zone_multipliers: dict[str, float] = field(default_factory=dict)

    # pipeline -> list of (min_confidence, tier), sorted descending by min_confidence
    tier_rules: dict[str, list[tuple[float, str]]] = field(default_factory=dict)

    # pipeline -> similarity_threshold
    similarity_thresholds: dict[str, float] = field(default_factory=dict)

    loaded_at: float = 0.0   # time.monotonic() at last successful load


class ConfigNotLoadedError(RuntimeError):
    """Raised if config is accessed before the first successful load."""
    pass


# ─────────────────────────────────────────────────────────────
# The loader / cache manager
# ─────────────────────────────────────────────────────────────

class ThresholdConfigLoader:
    """
    Owns the in-memory cache and knows when to refresh it.

    Usage:
        loader = ThresholdConfigLoader()
        await loader.ensure_fresh()
        threshold = loader.get_effective_threshold("violence", hour=17, zone_id="gym_east")

    A single instance of this class should be created at app startup
    (see config/thresholds.py:config_loader singleton below) and
    reused everywhere — NOT re-instantiated per request.
    """

    def __init__(self, ttl_seconds: int | None = None) -> None:
        self._cache = ThresholdConfigCache()
        self._ttl_seconds = ttl_seconds if ttl_seconds is not None else settings.config_cache_ttl_seconds

    @property
    def is_stale(self) -> bool:
        if self._cache.loaded_at == 0.0:
            return True   # never loaded
        return (time.monotonic() - self._cache.loaded_at) >= self._ttl_seconds

    async def ensure_fresh(self) -> None:
        """
        Refreshes the cache if it's stale or has never been loaded.
        Cheap no-op call if the cache is still fresh — safe to call
        at the start of every Layer 1 invocation without performance
        concern (it's just a time comparison in the common case).
        """
        if self.is_stale:
            await self._reload()

    async def force_reload(self) -> None:
        """
        Bypasses the TTL check entirely. Used by the PUT /config
        endpoint (Step 9) so that when an operator updates a
        threshold via the dashboard, the change is reflected
        immediately rather than waiting for the next TTL expiry.
        """
        await self._reload()

    async def _reload(self) -> None:
        new_cache = ThresholdConfigCache()

        async with db_session_scope() as session:
            # ── threshold_config ──
            result = await session.execute(select(ThresholdConfigRow))
            for row in result.scalars().all():
                new_cache.thresholds[row.pipeline] = (
                    row.base_threshold,
                    row.dedup_ttl_seconds,
                    row.escalation_delta,
                )

            # ── time_multiplier_config ──
            result = await session.execute(select(TimeMultiplierConfigRow))
            new_cache.time_multipliers = [
                (row.hour_start, row.hour_end, row.multiplier)
                for row in result.scalars().all()
            ]

            # ── zone_config ──
            result = await session.execute(select(ZoneConfigRow))
            for row in result.scalars().all():
                new_cache.zone_multipliers[row.zone_id] = row.risk_multiplier

            # ── tier_config ──
            result = await session.execute(
                select(TierConfigRow).order_by(TierConfigRow.min_confidence.desc())
            )
            for row in result.scalars().all():
                new_cache.tier_rules.setdefault(row.pipeline, []).append(
                    (row.min_confidence, row.tier)
                )

            # ── suppression_similarity_config ──
            result = await session.execute(select(SuppressionSimilarityConfigRow))
            for row in result.scalars().all():
                new_cache.similarity_thresholds[row.pipeline] = row.similarity_threshold

        new_cache.loaded_at = time.monotonic()
        self._cache = new_cache

    # ─────────────────────────────────────────────────────────
    # Public read accessors — used by the layers
    # ─────────────────────────────────────────────────────────

    def _require_loaded(self) -> None:
        if self._cache.loaded_at == 0.0:
            raise ConfigNotLoadedError(
                "Threshold config has not been loaded yet. "
                "Call `await loader.ensure_fresh()` before reading config "
                "(this happens automatically at FastAPI startup)."
            )

    def get_time_multiplier(self, hour: int) -> float:
        """
        Returns the multiplier for the given hour (0-23).
        Defaults to 1.0 (neutral) if no configured window covers
        this hour — a gap in config should never crash the pipeline,
        it should just mean "no adjustment."
        """
        self._require_loaded()
        for hour_start, hour_end, multiplier in self._cache.time_multipliers:
            if hour_start <= hour <= hour_end:
                return multiplier
        return 1.0

    def get_zone_multiplier(self, zone_id: str) -> float:
        """
        Defaults to 1.0 if the zone isn't configured — an unknown
        zone should not artificially inflate or deflate sensitivity.
        """
        self._require_loaded()
        return self._cache.zone_multipliers.get(zone_id, 1.0)

    def get_base_threshold(self, pipeline: str) -> float:
        self._require_loaded()
        if pipeline not in self._cache.thresholds:
            raise KeyError(
                f"No threshold_config row for pipeline '{pipeline}'. "
                f"Every pipeline must have a row — check db/init/seed.sql "
                f"or the threshold_config table directly."
            )
        return self._cache.thresholds[pipeline][0]

    def get_dedup_ttl(self, pipeline: str) -> int:
        self._require_loaded()
        return self._cache.thresholds[pipeline][1]

    def get_escalation_delta(self, pipeline: str) -> float:
        self._require_loaded()
        return self._cache.thresholds[pipeline][2]

    def get_effective_threshold(self, pipeline: str, hour: int, zone_id: str) -> float:
        """
        The core formula from architecture doc §3, Layer 1:

            effective_threshold = base_threshold(pipeline)
                                   × time_multiplier(hour)
                                   × zone_risk_multiplier(zone_id)

        This is the single function Layer 1 calls to get the number
        it compares incoming confidence against.
        """
        base = self.get_base_threshold(pipeline)
        time_mult = self.get_time_multiplier(hour)
        zone_mult = self.get_zone_multiplier(zone_id)
        return base * time_mult * zone_mult

    def get_tier(self, pipeline: str, effective_conf: float) -> str:
        """
        Walks tier_rules for this pipeline (pre-sorted descending by
        min_confidence at load time) and returns the first tier whose
        min_confidence the event qualifies for.

        Falls back to "LOW" if no rule matches — every alert that
        survives all four gates gets displayed somewhere, never silently
        dropped at the tiering stage.
        """
        self._require_loaded()
        rules = self._cache.tier_rules.get(pipeline, [])
        for min_confidence, tier in rules:
            if effective_conf >= min_confidence:
                return tier
        return "LOW"

    def get_similarity_threshold(self, pipeline: str) -> float:
        """
        Used by Layer 4B (semantic suppression). Defaults conservatively
        to 0.95 (suppress almost nothing) if a pipeline somehow has no
        configured value — fail toward showing alerts, not hiding them.
        """
        self._require_loaded()
        return self._cache.similarity_thresholds.get(pipeline, 0.95)


# ─────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────

# One loader instance shared across the whole application.
# main.py's startup event calls `await config_loader.ensure_fresh()`
# once before the app starts accepting requests (Step 9).
config_loader = ThresholdConfigLoader()