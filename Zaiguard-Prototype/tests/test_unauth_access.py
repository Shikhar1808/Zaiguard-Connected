"""
tests/test_unauth_access.py

Unit tests for the camera-schedule-based UnauthAccessClassifier.
No GPU or CV2 polygon calls involved — pure time + threshold logic.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from classifiers.unauth_access import UnauthAccessClassifier
from core.config_loader import CameraSchedule, ThresholdConfig, TimeWindow
from core.packets import SceneFeatures, TrackBox, TrackPacket


# ── fixtures ──────────────────────────────────────────────────────────────────

def make_schedule(
    camera_id: str = "cam_01",
    restricted: bool = True,
    allowed_windows: list[tuple[int, int]] = None,
) -> CameraSchedule:
    windows = [TimeWindow(start=s, end=e) for s, e in (allowed_windows or [])]
    return CameraSchedule(camera_id=camera_id, label=camera_id, restricted=restricted, allowed=windows)


def make_thresholds(
    min_frames: int = 3,
    score: float = 0.6,
    cooldown: float = 0.0,
) -> ThresholdConfig:
    return ThresholdConfig(
        unauth_min_frames=min_frames,
        unauth_score=score,
        unauth_cooldown_s=cooldown,
        unauth_global_cooldown_s=0.0,
        unauth_min_confidence=0.0,
    )


def make_track(track_id: int = 1, class_label: str = "person") -> TrackBox:
    return TrackBox(
        track_id=track_id,
        bbox=[10, 10, 100, 200],
        bbox_norm=[0.01, 0.01, 0.1, 0.2],
        score=0.9,
        class_id=0,
        class_label=class_label,
        centroid=[55, 105],
        centroid_norm=[0.055, 0.105],
        area_px=90 * 190,
        aspect_ratio=round(90 / 190, 3),
    )


def make_packet(
    tracks: list[TrackBox] | None = None,
    camera_id: str = "cam_01",
) -> TrackPacket:
    import numpy as np
    return TrackPacket(
        camera_id=camera_id,
        frame_id=1,
        timestamp=time.time(),
        frame=np.zeros((480, 640, 3), dtype="uint8"),
        frame_shape=[480, 640, 3],
        tracks=tracks or [make_track()],
        scene_features=SceneFeatures(track_count=1, person_count=1),
    )


def classify_n(clf: UnauthAccessClassifier, packet: TrackPacket, n: int):
    alerts = []
    for _ in range(n):
        alerts.extend(clf.classify(packet))
    return alerts


# ── time gate helpers ─────────────────────────────────────────────────────────

def patch_hour(h: int):
    """Context manager that freezes datetime.now().hour."""
    from datetime import datetime
    fake = datetime(2025, 1, 15, h, 30, 0)
    return patch("classifiers.unauth_access.datetime", wraps=datetime,
                 **{"now.return_value": fake})


# ── tests ─────────────────────────────────────────────────────────────────────

class TestTimeWindow:
    def test_normal_window(self):
        w = TimeWindow(start=8, end=18)
        assert w.is_active(8)
        assert w.is_active(12)
        assert w.is_active(18)
        assert not w.is_active(7)
        assert not w.is_active(19)

    def test_overnight_window(self):
        w = TimeWindow(start=22, end=6)
        assert w.is_active(22)
        assert w.is_active(0)
        assert w.is_active(3)
        assert w.is_active(6)
        assert not w.is_active(7)
        assert not w.is_active(21)

    def test_full_day(self):
        w = TimeWindow(start=0, end=23)
        for h in range(24):
            assert w.is_active(h)


class TestCameraSchedule:
    def test_unrestricted_always_allowed(self):
        s = make_schedule(restricted=False)
        for h in range(24):
            assert s.is_allowed_now(h)

    def test_no_windows_always_restricted(self):
        s = make_schedule(restricted=True, allowed_windows=[])
        for h in range(24):
            assert not s.is_allowed_now(h)

    def test_multiple_windows_or_logic(self):
        # Morning + afternoon, restricted at lunch and night
        s = make_schedule(allowed_windows=[(8, 11), (13, 18)])
        assert s.is_allowed_now(9)
        assert s.is_allowed_now(15)
        assert not s.is_allowed_now(12)   # exactly 12 IS in first window
        assert not s.is_allowed_now(19)
        assert not s.is_allowed_now(2)


class TestUnauthAccessClassifier:

    def _make_clf(self, **kwargs):
        sched = make_schedule(**{k: v for k, v in kwargs.items()
                                  if k in ("camera_id", "restricted", "allowed_windows")})
        thresh_kwargs = {k: v for k, v in kwargs.items()
                        if k in ("min_frames", "score", "cooldown")}
        clf = UnauthAccessClassifier([sched], make_thresholds(**thresh_kwargs))
        clf.setup()
        return clf

    def test_fires_during_restricted_hours(self):
        clf = self._make_clf(allowed_windows=[(8, 18)], min_frames=3, score=0.6)
        with patch_hour(2):   # 02:00 — outside allowed window
            alerts = classify_n(clf, make_packet(), 3)
        assert len(alerts) == 1
        assert alerts[0].event_type == "unauth_access"

    def test_no_alert_during_allowed_hours(self):
        clf = self._make_clf(allowed_windows=[(8, 18)], min_frames=3, score=0.6)
        with patch_hour(10):  # inside window
            alerts = classify_n(clf, make_packet(), 5)
        assert alerts == []

    def test_no_alert_before_window_fills(self):
        clf = self._make_clf(allowed_windows=[], min_frames=5, score=0.6)
        with patch_hour(2):
            alerts = classify_n(clf, make_packet(), 4)  # one short
        assert alerts == []

    def test_no_alert_for_non_person(self):
        clf = self._make_clf(allowed_windows=[], min_frames=3, score=0.6)
        car_track = make_track(class_label="car")
        with patch_hour(2):
            alerts = classify_n(clf, make_packet(tracks=[car_track]), 5)
        assert alerts == []

    def test_no_alert_unrestricted_camera(self):
        clf = self._make_clf(restricted=False, min_frames=3, score=0.6)
        with patch_hour(2):
            alerts = classify_n(clf, make_packet(), 5)
        assert alerts == []

    def test_no_alert_wrong_camera(self):
        sched = make_schedule(camera_id="cam_02")
        clf = UnauthAccessClassifier([sched], make_thresholds(min_frames=3))
        clf.setup()
        with patch_hour(2):
            # Packet is from cam_01; schedule is for cam_02
            alerts = classify_n(clf, make_packet(camera_id="cam_01"), 5)
        assert alerts == []

    def test_cooldown_suppresses_repeat(self):
        clf = self._make_clf(allowed_windows=[], min_frames=3, score=0.6, cooldown=60.0)
        with patch_hour(2):
            first  = classify_n(clf, make_packet(), 3)
            second = classify_n(clf, make_packet(), 3)
        assert len(first) == 1
        assert len(second) == 0

    def test_alert_meta_has_time_fields(self):
        clf = self._make_clf(allowed_windows=[], min_frames=3, score=0.6)
        with patch_hour(2):
            alerts = classify_n(clf, make_packet(), 3)
        assert len(alerts) == 1
        meta = alerts[0].meta
        assert meta.hour_of_day == 2
        assert meta.wall_time != ""
        assert meta.day_of_week != ""
        # camera_restricted lives on ThresholdVerdict, not AlertMeta
        assert meta.threshold_verdict.camera_restricted is True
        # v1.3 fields
        assert meta.severity in ("low", "medium", "high", "critical")
        assert meta.frame_width > 0
        assert meta.frame_height > 0

    def test_verdict_fields_populated(self):
        clf = self._make_clf(allowed_windows=[], min_frames=3, score=0.6)
        with patch_hour(2):
            alerts = classify_n(clf, make_packet(), 3)
        v = alerts[0].meta.threshold_verdict
        assert v.frames_evaluated == 3
        assert v.frames_in_violation == 3
        assert v.raw_score == 1.0
        assert v.passed is True
        assert v.camera_restricted is True

    def test_prune_stale_tracks(self):
        clf = self._make_clf(allowed_windows=[], min_frames=3, score=0.6)
        with patch_hour(2):
            classify_n(clf, make_packet(tracks=[make_track(track_id=99)]), 3)
        assert ("cam_01", 99) in clf._history
        clf.prune_stale_tracks(active_ids=set(), camera_id="cam_01")
        assert len(clf._history) == 0

    def test_overnight_window(self):
        # Overnight: allowed 22:00–06:00
        clf = self._make_clf(allowed_windows=[(22, 6)], min_frames=3, score=0.6)
        with patch_hour(3):    # inside overnight window → allowed, no alert
            alerts = classify_n(clf, make_packet(), 5)
        assert alerts == []
        clf2 = self._make_clf(allowed_windows=[(22, 6)], min_frames=3, score=0.6)
        with patch_hour(14):   # outside overnight window → restricted → alert
            alerts2 = classify_n(clf2, make_packet(), 3)
        assert len(alerts2) == 1

    def test_v13_metadata(self):
        """v1.3: track_duration, severity populated."""
        clf = self._make_clf(allowed_windows=[], min_frames=3, score=0.6)
        with patch_hour(2):
            alerts = classify_n(clf, make_packet(), 3)
        assert len(alerts) == 1
        # track duration should be non-negative
        assert alerts[0].meta.track_duration_s >= 0.0
        assert alerts[0].meta.track_first_seen_frame >= 0
        # severity is always set
        assert alerts[0].meta.severity in ("low", "medium", "high", "critical")

    def test_prune_cleans_track_first_seen(self):
        """v1.3: prune_stale_tracks also removes _track_first_seen entries."""
        clf = self._make_clf(allowed_windows=[], min_frames=3, score=0.6)
        with patch_hour(2):
            classify_n(clf, make_packet(tracks=[make_track(track_id=42)]), 3)
        assert ("cam_01", 42) in clf._track_first_seen
        clf.prune_stale_tracks(active_ids=set(), camera_id="cam_01")
        assert ("cam_01", 42) not in clf._track_first_seen
