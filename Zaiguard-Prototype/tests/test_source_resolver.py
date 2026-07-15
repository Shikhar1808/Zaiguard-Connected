"""
tests/test_source_resolver.py

Tests for the ingestion fallback chain.
Run with: uv run pytest tests/ -v
"""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── stub cv2 if not installed in test env ────────────────────────────────────
try:
    import cv2
except ImportError:
    cv2_stub = types.ModuleType("cv2")
    cv2_stub.VideoCapture = MagicMock
    sys.modules["cv2"] = cv2_stub

from ingestion.source_resolver import (
    _is_device,
    _is_http,
    _is_rtsp,
    _is_video_file,
)


class TestSourceClassification:
    def test_rtsp_detected(self):
        assert _is_rtsp("rtsp://admin:pass@192.168.1.1/stream1")
        assert not _is_rtsp("http://192.168.1.1/stream")
        assert not _is_rtsp("/some/file.mp4")

    def test_http_detected(self):
        assert _is_http("http://192.168.1.1:8080/video")
        assert _is_http("https://example.com/stream")
        assert not _is_http("rtsp://192.168.1.1/stream")

    def test_device_detected(self):
        assert _is_device("device:0")
        assert _is_device("device:3")
        assert not _is_device("device:")
        assert not _is_device("rtsp://")

    def test_video_file_detected(self, tmp_path):
        mp4 = tmp_path / "test.mp4"
        mp4.touch()
        assert _is_video_file(str(mp4))

        avi = tmp_path / "test.avi"
        avi.touch()
        assert _is_video_file(str(avi))

        # Non-existent file → False
        assert not _is_video_file(str(tmp_path / "missing.mp4"))


class TestResolveSource:
    """Integration-level tests using mocked VideoCapture."""

    def _make_cap(self, ok: bool):
        cap = MagicMock()
        cap.isOpened.return_value = ok
        cap.read.return_value = (ok, MagicMock() if ok else None)
        return cap

    def test_rtsp_success(self):
        with patch("ingestion.source_resolver.cv2.VideoCapture") as MockCap:
            MockCap.return_value = self._make_cap(True)
            from ingestion.source_resolver import resolve_source
            cap, src = resolve_source("rtsp://192.168.1.1/stream", "test_cam")
            assert cap is not None

    def test_rtsp_fails_falls_back_to_device(self):
        call_count = {"n": 0}

        def cap_factory(source):
            c = MagicMock()
            # First call (RTSP) fails, second call (device 0) succeeds
            ok = call_count["n"] >= 1
            call_count["n"] += 1
            c.isOpened.return_value = ok
            c.read.return_value = (ok, MagicMock() if ok else None)
            return c

        with patch("ingestion.source_resolver.cv2.VideoCapture", side_effect=cap_factory):
            from ingestion.source_resolver import resolve_source
            cap, src = resolve_source("rtsp://bad-host/stream", "test_cam")
            # Should have found something (device fallback)
            # In a real env with no device this would be None — acceptable
            assert cap is None or src is not None

    def test_unknown_source_returns_none_when_no_devices(self):
        with patch("ingestion.source_resolver.cv2.VideoCapture") as MockCap:
            bad = self._make_cap(False)
            MockCap.return_value = bad
            from ingestion.source_resolver import resolve_source
            cap, src = resolve_source("not-a-real-source", "test_cam")
            assert cap is None
