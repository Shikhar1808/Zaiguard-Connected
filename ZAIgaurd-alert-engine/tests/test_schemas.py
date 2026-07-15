"""
Tests for models/schemas.py
============================
These tests run with zero external services (no Redis, Postgres, Qdrant).
They validate that the Pydantic schemas correctly accept valid data,
reject invalid data, and that helper functions work as expected.

Run with:
    pytest tests/test_schemas.py -v
"""

import hashlib
from datetime import datetime, timezone, timedelta

import pytest
from pydantic import ValidationError

from models.schemas import (
    AlertEvent,
    AlertTier,
    DashboardAlert,
    OperatorAction,
    OperatorFeedback,
    Pipeline,
    RawDetectionEvent,
    SuppressionRule,
    build_alert_id,
)


# ─────────────────────────────────────────────────────────────
# Fixtures — reusable test objects
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def valid_raw_event() -> RawDetectionEvent:
    """A valid RawDetectionEvent covering the common case."""
    return RawDetectionEvent(
        pipeline=Pipeline.VIOLENCE,
        raw_confidence=0.85,
        camera_id="CAM_07",
        zone_id="gym_east",
        zone_label="Gym East",
        timestamp=datetime(2026, 6, 8, 17, 0, 0, tzinfo=timezone.utc),
        frame_ref="/clips/2026-06-08/CAM_07_17-00-00.mp4",
        involved_ids=[3, 7],
        pipeline_features={
            "inter_person_distance": 0.4,
            "relative_velocity": 2.1,
        },
    )


@pytest.fixture
def valid_alert_event(valid_raw_event) -> AlertEvent:
    """An AlertEvent built from the raw event above."""
    return AlertEvent(
        alert_id=build_alert_id(
            valid_raw_event.camera_id,
            valid_raw_event.timestamp,
            valid_raw_event.pipeline,
        ),
        source_event=valid_raw_event,
        effective_conf=valid_raw_event.raw_confidence,
        hour_of_day=valid_raw_event.timestamp.hour,
        day_of_week=valid_raw_event.timestamp.weekday(),
    )


# ─────────────────────────────────────────────────────────────
# RawDetectionEvent tests
# ─────────────────────────────────────────────────────────────

class TestRawDetectionEvent:

    def test_valid_event_constructs(self, valid_raw_event):
        assert valid_raw_event.pipeline == Pipeline.VIOLENCE
        assert valid_raw_event.camera_id == "CAM_07"

    def test_confidence_below_zero_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            RawDetectionEvent(
                pipeline=Pipeline.FIRE,
                raw_confidence=-0.1,
                camera_id="CAM_01",
            )
        assert "raw_confidence" in str(exc_info.value)

    def test_confidence_above_one_rejected(self):
        with pytest.raises(ValidationError):
            RawDetectionEvent(
                pipeline=Pipeline.FIRE,
                raw_confidence=1.01,
                camera_id="CAM_01",
            )

    def test_confidence_boundary_values_accepted(self):
        """0.0 and 1.0 are valid confidence values."""
        for conf in [0.0, 1.0]:
            event = RawDetectionEvent(
                pipeline=Pipeline.TRESPASSING,
                raw_confidence=conf,
                camera_id="CAM_05",
            )
            assert event.raw_confidence == conf

    def test_naive_datetime_gets_utc(self):
        """A timestamp without timezone info should be treated as UTC."""
        naive_ts = datetime(2026, 6, 8, 12, 0, 0)  # no tzinfo
        event = RawDetectionEvent(
            pipeline=Pipeline.DOG_ATTACK,
            raw_confidence=0.7,
            camera_id="CAM_03",
            timestamp=naive_ts,
        )
        assert event.timestamp.tzinfo is not None
        assert event.timestamp.tzinfo == timezone.utc

    def test_non_utc_datetime_converted_to_utc(self):
        """A non-UTC timezone should be converted to UTC."""
        from datetime import timezone as tz
        ist_offset = timedelta(hours=5, minutes=30)
        ist = tz(ist_offset)
        # 15:00 IST = 09:30 UTC
        ist_ts = datetime(2026, 6, 8, 15, 0, 0, tzinfo=ist)
        event = RawDetectionEvent(
            pipeline=Pipeline.VIOLENCE,
            raw_confidence=0.8,
            camera_id="CAM_07",
            timestamp=ist_ts,
        )
        assert event.timestamp.hour == 9
        assert event.timestamp.minute == 30

    def test_defaults_are_sensible(self):
        """Minimal construction should fill in safe defaults."""
        event = RawDetectionEvent(
            pipeline=Pipeline.ACCIDENT,
            raw_confidence=0.75,
            camera_id="CAM_10",
        )
        assert event.zone_id == "default"
        assert event.zone_label == "Unknown Zone"
        assert event.involved_ids == []
        assert event.pipeline_features == {}
        assert event.frame_ref is None

    def test_invalid_pipeline_string_rejected(self):
        with pytest.raises(ValidationError):
            RawDetectionEvent(
                pipeline="explosions",   # not in Pipeline enum
                raw_confidence=0.9,
                camera_id="CAM_01",
            )

    def test_all_valid_pipelines_accepted(self):
        for p in Pipeline:
            event = RawDetectionEvent(
                pipeline=p,
                raw_confidence=0.75,
                camera_id="CAM_01",
            )
            assert event.pipeline == p


# ─────────────────────────────────────────────────────────────
# AlertEvent tests
# ─────────────────────────────────────────────────────────────

class TestAlertEvent:

    def test_convenience_properties_delegate_to_source(self, valid_alert_event, valid_raw_event):
        """Properties on AlertEvent should transparently proxy to source_event."""
        assert valid_alert_event.pipeline == valid_raw_event.pipeline
        assert valid_alert_event.camera_id == valid_raw_event.camera_id
        assert valid_alert_event.zone_id == valid_raw_event.zone_id
        assert valid_alert_event.raw_confidence == valid_raw_event.raw_confidence

    def test_hour_of_day_is_correct(self, valid_alert_event):
        assert valid_alert_event.hour_of_day == 17

    def test_day_of_week_is_correct(self, valid_alert_event):
        # 2026-06-08 is a Monday (weekday() == 0)
        assert valid_alert_event.day_of_week == 0


# ─────────────────────────────────────────────────────────────
# DashboardAlert tests
# ─────────────────────────────────────────────────────────────

class TestDashboardAlert:

    def test_suppression_score_defaults_to_zero(self):
        alert = DashboardAlert(
            alert_id="abc123",
            tier=AlertTier.HIGH,
            pipeline=Pipeline.VIOLENCE,
            camera_id="CAM_07",
            zone_id="gym_east",
            zone_label="Gym East",
            raw_confidence=0.85,
            effective_conf=0.85,
            timestamp=datetime.now(timezone.utc),
            evidence_frame_ref=None,
            involved_ids=[],
        )
        assert alert.suppression_score == 0.0

    def test_suppression_score_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            DashboardAlert(
                alert_id="abc123",
                tier=AlertTier.HIGH,
                pipeline=Pipeline.VIOLENCE,
                camera_id="CAM_07",
                zone_id="gym_east",
                zone_label="Gym East",
                raw_confidence=0.85,
                effective_conf=0.85,
                timestamp=datetime.now(timezone.utc),
                evidence_frame_ref=None,
                involved_ids=[],
                suppression_score=1.5,  # invalid
            )


# ─────────────────────────────────────────────────────────────
# OperatorFeedback tests
# ─────────────────────────────────────────────────────────────

class TestOperatorFeedback:

    def test_valid_dismiss(self):
        fb = OperatorFeedback(alert_id="abc", action=OperatorAction.DISMISS)
        assert fb.permanent is False

    def test_permanent_dismiss(self):
        fb = OperatorFeedback(
            alert_id="abc",
            action=OperatorAction.DISMISS,
            permanent=True,
        )
        assert fb.permanent is True

    def test_permanent_confirm_is_invalid(self):
        """permanent=True makes no sense on a CONFIRM action."""
        with pytest.raises(ValidationError) as exc_info:
            OperatorFeedback(
                alert_id="abc",
                action=OperatorAction.CONFIRM,
                permanent=True,
            )
        assert "permanent" in str(exc_info.value)

    def test_confirm_without_permanent_is_valid(self):
        fb = OperatorFeedback(alert_id="abc", action=OperatorAction.CONFIRM)
        assert fb.action == OperatorAction.CONFIRM


# ─────────────────────────────────────────────────────────────
# SuppressionRule tests
# ─────────────────────────────────────────────────────────────

class TestSuppressionRule:

    def test_valid_rule_with_time_window(self):
        rule = SuppressionRule(
            camera_id="CAM_07",
            pipeline=Pipeline.VIOLENCE,
            hour_start=16,
            hour_end=19,
            days_mask=31,   # weekdays
        )
        assert rule.hour_start == 16

    def test_hour_start_without_hour_end_rejected(self):
        with pytest.raises(ValidationError):
            SuppressionRule(
                camera_id="CAM_07",
                pipeline=Pipeline.VIOLENCE,
                hour_start=16,
                # hour_end missing
            )

    def test_hour_start_greater_than_end_rejected(self):
        with pytest.raises(ValidationError):
            SuppressionRule(
                camera_id="CAM_07",
                pipeline=Pipeline.VIOLENCE,
                hour_start=20,
                hour_end=16,  # invalid: start > end
            )

    def test_rule_without_time_window_is_valid(self):
        """A rule with no time window suppresses at all times."""
        rule = SuppressionRule(
            camera_id="CAM_07",
            pipeline=Pipeline.VIOLENCE,
        )
        assert rule.hour_start is None
        assert rule.hour_end is None


# ─────────────────────────────────────────────────────────────
# build_alert_id tests
# ─────────────────────────────────────────────────────────────

class TestBuildAlertId:

    def test_same_inputs_produce_same_id(self):
        """Determinism is the entire point — verify it explicitly."""
        ts = datetime(2026, 6, 8, 17, 0, 0, tzinfo=timezone.utc)
        id1 = build_alert_id("CAM_07", ts, Pipeline.VIOLENCE)
        id2 = build_alert_id("CAM_07", ts, Pipeline.VIOLENCE)
        assert id1 == id2

    def test_different_camera_produces_different_id(self):
        ts = datetime(2026, 6, 8, 17, 0, 0, tzinfo=timezone.utc)
        id1 = build_alert_id("CAM_07", ts, Pipeline.VIOLENCE)
        id2 = build_alert_id("CAM_08", ts, Pipeline.VIOLENCE)
        assert id1 != id2

    def test_different_timestamp_produces_different_id(self):
        ts1 = datetime(2026, 6, 8, 17, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2026, 6, 8, 17, 0, 1, tzinfo=timezone.utc)
        id1 = build_alert_id("CAM_07", ts1, Pipeline.VIOLENCE)
        id2 = build_alert_id("CAM_07", ts2, Pipeline.VIOLENCE)
        assert id1 != id2

    def test_different_pipeline_produces_different_id(self):
        ts = datetime(2026, 6, 8, 17, 0, 0, tzinfo=timezone.utc)
        id1 = build_alert_id("CAM_07", ts, Pipeline.VIOLENCE)
        id2 = build_alert_id("CAM_07", ts, Pipeline.FIRE)
        assert id1 != id2

    def test_id_is_32_hex_chars(self):
        ts = datetime(2026, 6, 8, 17, 0, 0, tzinfo=timezone.utc)
        alert_id = build_alert_id("CAM_07", ts, Pipeline.VIOLENCE)
        assert len(alert_id) == 32
        assert all(c in "0123456789abcdef" for c in alert_id)