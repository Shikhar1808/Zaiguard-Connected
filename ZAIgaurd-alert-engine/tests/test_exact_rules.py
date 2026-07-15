"""
Tests for layers/suppression/exact_rules.py
=============================================
These tests run against your LIVE local Postgres container.
Docker must be running with zaiguard_postgres healthy before
running this file.

    docker compose ps   ← should show zaiguard_postgres healthy
    pytest tests/test_exact_rules.py -v

WHY LIVE POSTGRES (not mocked)
--------------------------------
Layer 4A's entire job is a SQL query. Mocking the database would
mean testing that SQLAlchemy calls the right methods on a mock
object — it would prove nothing about whether the query is correct,
whether the index is used, or whether the bitmask arithmetic works
in real Postgres. The only meaningful test is against a real database.

TEST ISOLATION
---------------
Each test class uses a pytest fixture that:
1. Inserts the suppression rules it needs before the test
2. Deletes them after the test (via RETURNING id + delete)

This means tests are fully isolated — they don't depend on each
other's state and can run in any order.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio

from config.database import db_session_scope
from config.thresholds import ThresholdConfigCache, ThresholdConfigLoader
from layers.suppression.exact_rules import (
    ExactRuleResult,
    run_exact_rule_check,
    write_suppression_rule,
)
from models.schemas import AlertEvent, Pipeline, RawDetectionEvent


# ─────────────────────────────────────────────────────────────
# Pytest config — all tests in this file are async
# ─────────────────────────────────────────────────────────────

pytestmark = pytest.mark.asyncio


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def make_alert_event(
    pipeline: Pipeline = Pipeline.VIOLENCE,
    camera_id: str = "CAM_TEST_07",
    zone_id: str = "gym_east",
    hour: int = 17,
    day_of_week: int = 0,       # Monday
    confidence: float = 0.85,
) -> AlertEvent:
    """
    Builds a test AlertEvent. camera_id prefixed with CAM_TEST_ to
    avoid any accidental collision with real rules in the database.
    """
    raw = RawDetectionEvent(
        pipeline=pipeline,
        raw_confidence=confidence,
        camera_id=camera_id,
        zone_id=zone_id,
        zone_label="Gym East",
        timestamp=datetime(2026, 6, 8, hour, 0, 0, tzinfo=timezone.utc),
    )
    return AlertEvent(
        alert_id=f"test_{uuid.uuid4().hex[:8]}",
        source_event=raw,
        effective_conf=confidence,
        hour_of_day=hour,
        day_of_week=day_of_week,
    )


async def insert_rule(
    session,
    camera_id: str = "CAM_TEST_07",
    pipeline: str = "violence",
    zone_id: str | None = "gym_east",
    hour_start: int | None = None,
    hour_end: int | None = None,
    days_mask: int | None = None,
    expires_at: datetime | None = None,
    source: str = "manual",
) -> str:
    """Inserts a rule directly and returns its UUID."""
    from sqlalchemy import text
    result = await session.execute(
        text("""
            INSERT INTO suppression_rules
                (camera_id, pipeline, zone_id, hour_start, hour_end,
                 days_mask, expires_at, source, created_at)
            VALUES
                (:camera_id, :pipeline, :zone_id, :hour_start, :hour_end,
                 :days_mask, :expires_at, :source, :created_at)
            RETURNING id::text
        """),
        {
            "camera_id":  camera_id,
            "pipeline":   pipeline,
            "zone_id":    zone_id,
            "hour_start": hour_start,
            "hour_end":   hour_end,
            "days_mask":  days_mask,
            "expires_at": expires_at,
            "source":     source,
            "created_at": datetime.now(timezone.utc),
        },
    )
    await session.commit()
    return result.fetchone()[0]


async def delete_rule(session, rule_id: str) -> None:
    """
    Cleans up a test rule by ID.

    Uses CAST(:id AS uuid) rather than :id::uuid because asyncpg's
    parameter parser sees ':id' as a named param then chokes on '::uuid'
    immediately after — it's a known asyncpg limitation with inline casts
    on named parameters. CAST(...) is standard SQL and works everywhere.
    """
    from sqlalchemy import text
    await session.execute(
        text("DELETE FROM suppression_rules WHERE id = CAST(:id AS uuid)"),
        {"id": rule_id},
    )
    await session.commit()


# ─────────────────────────────────────────────────────────────
# No matching rule — event should pass through
# ─────────────────────────────────────────────────────────────

class TestNoMatchingRule:

    async def test_no_rules_at_all_passes_through(self):
        """With a clean camera ID that has no rules, result is not suppressed."""
        event = make_alert_event(camera_id="CAM_TEST_NORULES")
        async with db_session_scope() as session:
            result = await run_exact_rule_check(event, session)

        assert result.suppressed is False
        assert result.matched_rule_id is None

    async def test_rule_for_different_camera_does_not_match(self):
        async with db_session_scope() as session:
            rule_id = await insert_rule(session, camera_id="CAM_TEST_OTHER")

        event = make_alert_event(camera_id="CAM_TEST_07")
        async with db_session_scope() as session:
            result = await run_exact_rule_check(event, session)
            await delete_rule(session, rule_id)

        assert result.suppressed is False

    async def test_rule_for_different_pipeline_does_not_match(self):
        async with db_session_scope() as session:
            rule_id = await insert_rule(session, pipeline="fire")

        event = make_alert_event(pipeline=Pipeline.VIOLENCE)
        async with db_session_scope() as session:
            result = await run_exact_rule_check(event, session)
            await delete_rule(session, rule_id)

        assert result.suppressed is False


# ─────────────────────────────────────────────────────────────
# Basic rule matching
# ─────────────────────────────────────────────────────────────

class TestBasicRuleMatching:

    async def test_rule_with_no_time_or_day_constraints_matches(self):
        """A rule with NULL hour and NULL days_mask matches at any time."""
        async with db_session_scope() as session:
            rule_id = await insert_rule(
                session,
                camera_id="CAM_TEST_07",
                pipeline="violence",
                hour_start=None,
                hour_end=None,
                days_mask=None,
            )

        event = make_alert_event()
        async with db_session_scope() as session:
            result = await run_exact_rule_check(event, session)
            await delete_rule(session, rule_id)

        assert result.suppressed is True
        assert result.matched_rule_id == rule_id

    async def test_matched_rule_id_returned_correctly(self):
        async with db_session_scope() as session:
            rule_id = await insert_rule(session)

        event = make_alert_event()
        async with db_session_scope() as session:
            result = await run_exact_rule_check(event, session)
            await delete_rule(session, rule_id)

        assert result.matched_rule_id == rule_id


# ─────────────────────────────────────────────────────────────
# Hour window matching
# ─────────────────────────────────────────────────────────────

class TestHourWindowMatching:

    async def test_event_inside_hour_window_is_suppressed(self):
        """Rule covers 16:00–19:00, event at 17:00 → suppressed."""
        async with db_session_scope() as session:
            rule_id = await insert_rule(
                session, hour_start=16, hour_end=19
            )

        event = make_alert_event(hour=17)
        async with db_session_scope() as session:
            result = await run_exact_rule_check(event, session)
            await delete_rule(session, rule_id)

        assert result.suppressed is True

    async def test_event_at_hour_window_start_boundary(self):
        """Rule covers 16:00–19:00, event at 16:00 → suppressed (inclusive)."""
        async with db_session_scope() as session:
            rule_id = await insert_rule(
                session, hour_start=16, hour_end=19
            )

        event = make_alert_event(hour=16)
        async with db_session_scope() as session:
            result = await run_exact_rule_check(event, session)
            await delete_rule(session, rule_id)

        assert result.suppressed is True

    async def test_event_at_hour_window_end_boundary(self):
        """Rule covers 16:00–19:00, event at 19:00 → suppressed (inclusive)."""
        async with db_session_scope() as session:
            rule_id = await insert_rule(
                session, hour_start=16, hour_end=19
            )

        event = make_alert_event(hour=19)
        async with db_session_scope() as session:
            result = await run_exact_rule_check(event, session)
            await delete_rule(session, rule_id)

        assert result.suppressed is True

    async def test_event_before_hour_window_is_not_suppressed(self):
        """Rule covers 16:00–19:00, event at 15:00 → not suppressed."""
        async with db_session_scope() as session:
            rule_id = await insert_rule(
                session, hour_start=16, hour_end=19
            )

        event = make_alert_event(hour=15)
        async with db_session_scope() as session:
            result = await run_exact_rule_check(event, session)
            await delete_rule(session, rule_id)

        assert result.suppressed is False

    async def test_event_after_hour_window_is_not_suppressed(self):
        """Rule covers 16:00–19:00, event at 20:00 → not suppressed."""
        async with db_session_scope() as session:
            rule_id = await insert_rule(
                session, hour_start=16, hour_end=19
            )

        event = make_alert_event(hour=20)
        async with db_session_scope() as session:
            result = await run_exact_rule_check(event, session)
            await delete_rule(session, rule_id)

        assert result.suppressed is False


# ─────────────────────────────────────────────────────────────
# Day of week bitmask matching
# ─────────────────────────────────────────────────────────────

class TestDayMaskMatching:

    async def test_weekday_rule_matches_on_monday(self):
        """
        days_mask=31 = Mon(1)+Tue(2)+Wed(4)+Thu(8)+Fri(16) = weekdays.
        Event on Monday (day_of_week=0) → suppressed.
        """
        async with db_session_scope() as session:
            rule_id = await insert_rule(session, days_mask=31)

        event = make_alert_event(day_of_week=0)  # Monday
        async with db_session_scope() as session:
            result = await run_exact_rule_check(event, session)
            await delete_rule(session, rule_id)

        assert result.suppressed is True

    async def test_weekday_rule_does_not_match_on_saturday(self):
        """days_mask=31 (weekdays), event on Saturday (day=5) → not suppressed."""
        async with db_session_scope() as session:
            rule_id = await insert_rule(session, days_mask=31)

        event = make_alert_event(day_of_week=5)  # Saturday
        async with db_session_scope() as session:
            result = await run_exact_rule_check(event, session)
            await delete_rule(session, rule_id)

        assert result.suppressed is False

    async def test_weekend_rule_matches_on_sunday(self):
        """days_mask=96 = Sat(32)+Sun(64). Event on Sunday (day=6) → suppressed."""
        async with db_session_scope() as session:
            rule_id = await insert_rule(session, days_mask=96)

        event = make_alert_event(day_of_week=6)  # Sunday
        async with db_session_scope() as session:
            result = await run_exact_rule_check(event, session)
            await delete_rule(session, rule_id)

        assert result.suppressed is True

    async def test_single_day_rule_only_matches_that_day(self):
        """days_mask=4 = Wednesday only. Event on Tuesday → not suppressed."""
        async with db_session_scope() as session:
            rule_id = await insert_rule(session, days_mask=4)  # Wednesday only

        event = make_alert_event(day_of_week=1)  # Tuesday
        async with db_session_scope() as session:
            result = await run_exact_rule_check(event, session)
            await delete_rule(session, rule_id)

        assert result.suppressed is False

    async def test_single_day_rule_matches_that_day(self):
        """days_mask=4 = Wednesday. Event on Wednesday (day=2) → suppressed."""
        async with db_session_scope() as session:
            rule_id = await insert_rule(session, days_mask=4)  # Wednesday

        event = make_alert_event(day_of_week=2)  # Wednesday
        async with db_session_scope() as session:
            result = await run_exact_rule_check(event, session)
            await delete_rule(session, rule_id)

        assert result.suppressed is True


# ─────────────────────────────────────────────────────────────
# Combined hour + day matching
# ─────────────────────────────────────────────────────────────

class TestCombinedConstraints:

    async def test_matching_hour_and_day_suppresses(self):
        """Basketball practice: weekdays 16-19. Monday 17:00 → suppressed."""
        async with db_session_scope() as session:
            rule_id = await insert_rule(
                session,
                hour_start=16, hour_end=19,
                days_mask=31,   # weekdays
            )

        event = make_alert_event(hour=17, day_of_week=0)
        async with db_session_scope() as session:
            result = await run_exact_rule_check(event, session)
            await delete_rule(session, rule_id)

        assert result.suppressed is True

    async def test_matching_hour_wrong_day_does_not_suppress(self):
        """Weekdays 16-19 rule. Saturday 17:00 → not suppressed."""
        async with db_session_scope() as session:
            rule_id = await insert_rule(
                session,
                hour_start=16, hour_end=19,
                days_mask=31,
            )

        event = make_alert_event(hour=17, day_of_week=5)  # Saturday
        async with db_session_scope() as session:
            result = await run_exact_rule_check(event, session)
            await delete_rule(session, rule_id)

        assert result.suppressed is False

    async def test_wrong_hour_matching_day_does_not_suppress(self):
        """Weekdays 16-19 rule. Monday 20:00 → not suppressed."""
        async with db_session_scope() as session:
            rule_id = await insert_rule(
                session,
                hour_start=16, hour_end=19,
                days_mask=31,
            )

        event = make_alert_event(hour=20, day_of_week=0)
        async with db_session_scope() as session:
            result = await run_exact_rule_check(event, session)
            await delete_rule(session, rule_id)

        assert result.suppressed is False


# ─────────────────────────────────────────────────────────────
# Rule expiry
# ─────────────────────────────────────────────────────────────

class TestRuleExpiry:

    async def test_permanent_rule_suppresses(self):
        """expires_at=NULL means permanent — should always suppress."""
        async with db_session_scope() as session:
            rule_id = await insert_rule(session, expires_at=None)

        event = make_alert_event()
        async with db_session_scope() as session:
            result = await run_exact_rule_check(event, session)
            await delete_rule(session, rule_id)

        assert result.suppressed is True

    async def test_future_expiry_rule_suppresses(self):
        """Rule expires in 24 hours — should still suppress now."""
        expires = datetime.now(timezone.utc) + timedelta(hours=24)
        async with db_session_scope() as session:
            rule_id = await insert_rule(session, expires_at=expires)

        event = make_alert_event()
        async with db_session_scope() as session:
            result = await run_exact_rule_check(event, session)
            await delete_rule(session, rule_id)

        assert result.suppressed is True

    async def test_expired_rule_does_not_suppress(self):
        """Rule expired 1 hour ago — should NOT suppress."""
        expires = datetime.now(timezone.utc) - timedelta(hours=1)
        async with db_session_scope() as session:
            rule_id = await insert_rule(session, expires_at=expires)

        event = make_alert_event()
        async with db_session_scope() as session:
            result = await run_exact_rule_check(event, session)
            await delete_rule(session, rule_id)

        assert result.suppressed is False

    async def test_just_expired_rule_does_not_suppress(self):
        """Rule expired 1 second ago — boundary case."""
        expires = datetime.now(timezone.utc) - timedelta(seconds=1)
        async with db_session_scope() as session:
            rule_id = await insert_rule(session, expires_at=expires)

        event = make_alert_event()
        async with db_session_scope() as session:
            result = await run_exact_rule_check(event, session)
            await delete_rule(session, rule_id)

        assert result.suppressed is False


# ─────────────────────────────────────────────────────────────
# Multiple rules — first match wins
# ─────────────────────────────────────────────────────────────

class TestMultipleRules:

    async def test_one_matching_among_multiple_suppresses(self):
        """
        Three rules: one for wrong camera, one expired, one valid.
        Only the valid one should match.
        """
        async with db_session_scope() as session:
            wrong_cam = await insert_rule(session, camera_id="CAM_TEST_WRONG")
            expired    = await insert_rule(
                session,
                expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
            )
            valid      = await insert_rule(session)  # matches everything

        event = make_alert_event()
        async with db_session_scope() as session:
            result = await run_exact_rule_check(event, session)
            await delete_rule(session, wrong_cam)
            await delete_rule(session, expired)
            await delete_rule(session, valid)

        assert result.suppressed is True
        assert result.matched_rule_id == valid

    async def test_no_matching_among_multiple_passes(self):
        """Two rules, both wrong camera — event should pass through."""
        async with db_session_scope() as session:
            r1 = await insert_rule(session, camera_id="CAM_TEST_A")
            r2 = await insert_rule(session, camera_id="CAM_TEST_B")

        event = make_alert_event(camera_id="CAM_TEST_07")
        async with db_session_scope() as session:
            result = await run_exact_rule_check(event, session)
            await delete_rule(session, r1)
            await delete_rule(session, r2)

        assert result.suppressed is False


# ─────────────────────────────────────────────────────────────
# write_suppression_rule — the write path
# ─────────────────────────────────────────────────────────────

class TestWriteSuppressionRule:

    async def test_write_creates_rule_that_can_be_matched(self):
        """
        Write a rule then immediately check it suppresses
        the corresponding event — full round-trip test.
        """
        expires = datetime.now(timezone.utc) + timedelta(hours=24)

        async with db_session_scope() as session:
            rule_id = await write_suppression_rule(
                camera_id="CAM_TEST_WRITE",
                pipeline="violence",
                zone_id="gym_east",
                hour_start=None,
                hour_end=None,
                days_mask=None,
                expires_at=expires,
                source="manual",
                session=session,
            )

        assert rule_id is not None

        event = make_alert_event(camera_id="CAM_TEST_WRITE")
        async with db_session_scope() as session:
            result = await run_exact_rule_check(event, session)
            await delete_rule(session, rule_id)

        assert result.suppressed is True
        assert result.matched_rule_id == rule_id

    async def test_write_returns_valid_uuid(self):
        async with db_session_scope() as session:
            rule_id = await write_suppression_rule(
                camera_id="CAM_TEST_UUID",
                pipeline="fire",
                zone_id=None,
                hour_start=None,
                hour_end=None,
                days_mask=None,
                expires_at=None,
                source="auto_promoted",
                session=session,
            )

        assert len(rule_id) == 36        # UUID format: 8-4-4-4-12
        assert rule_id.count("-") == 4

        async with db_session_scope() as session:
            await delete_rule(session, rule_id)