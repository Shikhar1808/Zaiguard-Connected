"""
tests/test_alert_engine_client.py

Tests for postproc/alert_engine_client.py.
"""

from unittest.mock import patch, MagicMock
import pytest
from core.packets import ConfirmedAlert, AlertMeta, SceneFeatures, ThresholdVerdict
from postproc.alert_engine_client import AlertEngineClient

def test_map_to_raw_event():
    client = AlertEngineClient(base_url="http://localhost:8000")
    scene = SceneFeatures(person_count=1, track_count=1)
    verdict = ThresholdVerdict(
        window_size=7,
        frames_in_violation=5,
        frames_evaluated=7,
        raw_score=0.71,
        threshold=0.6,
        passed=True,
        cooldown_active=False,
        camera_restricted=True,
        schedule_summary="allowed 08:00-18:00"
    )
    meta = AlertMeta(
        bbox=[100, 100, 200, 200],
        centroid=[150, 150],
        centroid_norm=[0.23, 0.31],
        aspect_ratio=1.0,
        hour_of_day=15,
        day_of_week="Tuesday",
        violation_history=[True] * 7,
        threshold_verdict=verdict,
        class_label="person",
        camera_id="cam_01",
        camera_label="Test Cam",
        detection_score=0.9,
        severity="critical",
        scene=scene,
        frame_width=640,
        frame_height=480,
        track_duration_s=10.0,
        track_first_seen_frame=1,
        wall_time="2026-07-14T21:15:00Z",
        schedule_summary="allowed 00:00-23:00",
        bbox_norm=[0.1, 0.1, 0.5, 0.5],
        area_px=100
    )
    alert = ConfirmedAlert(
        alert_id="test_alert_id_123",
        camera_id="cam_01",
        timestamp=1784030100.0,
        ts_iso="2026-07-14T21:15:00Z",
        event_type="unauth_access",
        track_id=1,
        confidence=0.95,
        zone_id="cam_01",
        clip_path="clips/test.mp4",
        snapshot_path="snapshots/test.jpg",
        meta=meta
    )
    
    payload = client._map_to_raw_event(alert)
    
    assert payload["pipeline"] == "unauth_access"
    assert payload["raw_confidence"] == 0.95
    assert payload["camera_id"] == "cam_01"
    assert payload["zone_id"] == "cam_01"
    assert payload["zone_label"] == "Test Cam"
    assert payload["frame_ref"] == "snapshots/test.jpg"
    assert payload["involved_ids"] == [1]
    assert payload["pipeline_features"]["severity"] == "critical"
    assert payload["pipeline_features"]["person_count"] == 1
    assert payload["pipeline_features"]["track_duration_s"] == 10.0
    assert payload["pipeline_features"]["frame_shape"] == [480, 640]

@patch("postproc.alert_engine_client.requests.post")
def test_send_success(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"alert": {"alert_id": "ae_alert_id_123", "tier": "CRITICAL"}}
    mock_post.return_value = mock_resp
    
    client = AlertEngineClient(base_url="http://localhost:8000")
    scene = SceneFeatures(person_count=1, track_count=1)
    verdict = ThresholdVerdict(
        window_size=7,
        frames_in_violation=5,
        frames_evaluated=7,
        raw_score=0.71,
        threshold=0.6,
        passed=True,
        cooldown_active=False,
        camera_restricted=True,
        schedule_summary="allowed 08:00-18:00"
    )
    meta = AlertMeta(
        bbox=[100, 100, 200, 200],
        centroid=[150, 150],
        centroid_norm=[0.23, 0.31],
        aspect_ratio=1.0,
        hour_of_day=15,
        day_of_week="Tuesday",
        violation_history=[True] * 7,
        threshold_verdict=verdict,
        class_label="person",
        camera_id="cam_01",
        camera_label="Test Cam",
        detection_score=0.9,
        severity="critical",
        scene=scene,
        frame_width=640,
        frame_height=480,
        track_duration_s=10.0,
        track_first_seen_frame=1,
        wall_time="2026-07-14T21:15:00Z",
        schedule_summary="allowed 00:00-23:00",
        bbox_norm=[0.1, 0.1, 0.5, 0.5],
        area_px=100
    )
    alert = ConfirmedAlert(
        alert_id="test_alert_id_123",
        camera_id="cam_01",
        timestamp=1784030100.0,
        ts_iso="2026-07-14T21:15:00Z",
        event_type="unauth_access",
        track_id=1,
        confidence=0.95,
        zone_id="cam_01",
        clip_path="clips/test.mp4",
        snapshot_path="snapshots/test.jpg",
        meta=meta
    )
    
    client._send_sync(alert)
    
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert args[0] == "http://localhost:8000/events"
    assert kwargs["json"]["pipeline"] == "unauth_access"
