"""
tests/test_config_loader.py

Tests for AppConfig validation, in particular:
  - at least 1 enabled camera is required (no upper limit otherwise)
  - schedules must reference a known camera_id
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.config_loader import AppConfig, CameraConfig, CameraSchedule


def make_camera(camera_id="cam_01", enabled=True) -> CameraConfig:
    return CameraConfig(camera_id=camera_id, source="device:0", enabled=enabled)


class TestMinimumCameraValidation:

    def test_zero_cameras_rejected(self):
        with pytest.raises(ValidationError, match="No enabled cameras"):
            AppConfig(cameras=[])

    def test_all_disabled_cameras_rejected(self):
        with pytest.raises(ValidationError, match="No enabled cameras"):
            AppConfig(cameras=[make_camera(enabled=False)])

    def test_one_enabled_camera_accepted(self):
        cfg = AppConfig(cameras=[make_camera(enabled=True)])
        assert len(cfg.cameras) == 1

    def test_many_cameras_accepted_no_upper_limit(self):
        cameras = [make_camera(camera_id=f"cam_{i:02d}") for i in range(50)]
        cfg = AppConfig(cameras=cameras)
        assert len(cfg.cameras) == 50

    def test_mix_of_enabled_and_disabled_accepted_if_one_enabled(self):
        cameras = [
            make_camera("cam_01", enabled=False),
            make_camera("cam_02", enabled=False),
            make_camera("cam_03", enabled=True),
        ]
        cfg = AppConfig(cameras=cameras)
        assert len(cfg.cameras) == 3


class TestScheduleCameraReferenceValidation:

    def test_schedule_for_unknown_camera_rejected(self):
        with pytest.raises(ValidationError, match="unknown camera_id"):
            AppConfig(
                cameras=[make_camera("cam_01")],
                schedules=[CameraSchedule(camera_id="cam_99")],
            )

    def test_schedule_for_known_camera_accepted(self):
        cfg = AppConfig(
            cameras=[make_camera("cam_01")],
            schedules=[CameraSchedule(camera_id="cam_01")],
        )
        assert len(cfg.schedules) == 1

    def test_multiple_cameras_multiple_schedules(self):
        cameras = [make_camera("cam_01"), make_camera("cam_02"), make_camera("cam_03")]
        schedules = [
            CameraSchedule(camera_id="cam_01"),
            CameraSchedule(camera_id="cam_02"),
        ]
        cfg = AppConfig(cameras=cameras, schedules=schedules)
        assert len(cfg.cameras) == 3
        assert len(cfg.schedules) == 2

    def test_schedule_for_unconfigured_camera_defaults_unrestricted(self):
        cfg = AppConfig(cameras=[make_camera("cam_01")])
        sched = cfg.schedule_for("cam_99")
        assert sched.restricted is False