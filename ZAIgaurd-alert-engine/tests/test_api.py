"""
Tests for main.py — FastAPI Application Endpoints
=================================================
Tests the HTTP interface of the ZaiGuard Alert Engine against live
Postgres, Redis, and Qdrant services using httpx.AsyncClient.

End points tested:
  - GET  /              (Root info)
  - GET  /health        (Dependencies status)
  - GET  /config        (Threshold config dump)
  - PUT  /config/reload (Force config refresh)
  - POST /events        (Pipeline execution: pass vs drop)
  - POST /feedback      (Operator action: confirm & dismiss)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
import pytest
import pytest_asyncio
import httpx

from config.database import db_session_scope
from main import app
from models.schemas import AlertTier, OperatorAction, Pipeline


@pytest_asyncio.fixture(scope="session")
async def async_client():
    """
    Session-scoped httpx AsyncClient wrapped inside FastAPI's lifespan context manager
    so startup (config loading, outbox worker) and shutdown cleanups run properly.
    """
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield client


@pytest_asyncio.fixture(autouse=True)
async def cleanup_redis_api_tests():
    """
    Clears test dedup keys before and after each test.
    """
    from config.database import get_redis_client
    redis = get_redis_client()
    try:
        keys = await redis.keys("CAM_TEST_API_*")
        if keys:
            await redis.delete(*keys)
    finally:
        await redis.aclose()


async def cleanup_alert_log(alert_id: str) -> None:
    from sqlalchemy import text
    async with db_session_scope() as session:
        await session.execute(
            text("DELETE FROM alert_log WHERE alert_id = :id"),
            {"id": alert_id},
        )


async def cleanup_suppression_rule_by_camera(camera_id: str) -> None:
    from sqlalchemy import text
    async with db_session_scope() as session:
        await session.execute(
            text("DELETE FROM suppression_rules WHERE camera_id = :cam"),
            {"cam": camera_id},
        )


class TestSystemEndpoints:

    @pytest.mark.asyncio
    async def test_root_endpoint(self, async_client: httpx.AsyncClient):
        response = await async_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "ZaiGuard Alert Engine"
        assert data["version"] == "0.1.0"

    @pytest.mark.asyncio
    async def test_health_check(self, async_client: httpx.AsyncClient):
        response = await async_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["services"]["postgres"] is True
        assert data["services"]["redis"] is True
        assert data["services"]["qdrant"] is True

    @pytest.mark.asyncio
    async def test_get_config(self, async_client: httpx.AsyncClient):
        response = await async_client.get("/config")
        assert response.status_code == 200
        data = response.json()
        assert "thresholds" in data
        assert "time_multipliers" in data
        assert "zone_multipliers" in data
        assert "tier_rules" in data
        assert "violence" in data["thresholds"]

    @pytest.mark.asyncio
    async def test_reload_config(self, async_client: httpx.AsyncClient):
        response = await async_client.put("/config/reload")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "reloaded"


class TestEventsEndpoint:

    @pytest.mark.asyncio
    async def test_post_event_high_confidence_creates_alert(self, async_client: httpx.AsyncClient):
        cam = f"CAM_TEST_API_PASS_{uuid.uuid4().hex[:6]}"
        payload = {
            "pipeline": "violence",
            "raw_confidence": 0.95,
            "camera_id": cam,
            "zone_id": "default",
            "zone_label": "Main Entrance",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "frame_ref": "/clips/api_test.mp4",
            "involved_ids": [10, 11]
        }
        response = await async_client.post("/events", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["alert"] is not None
        alert = data["alert"]
        try:
            assert alert["camera_id"] == cam
            assert alert["pipeline"] == "violence"
            assert alert["tier"] == "CRITICAL"
        finally:
            await cleanup_alert_log(alert["alert_id"])

    @pytest.mark.asyncio
    async def test_post_event_low_confidence_returns_null_alert(self, async_client: httpx.AsyncClient):
        cam = f"CAM_TEST_API_DROP_{uuid.uuid4().hex[:6]}"
        payload = {
            "pipeline": "trespassing",
            "raw_confidence": 0.25,
            "camera_id": cam,
            "zone_id": "default",
            "zone_label": "Side Alley",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        response = await async_client.post("/events", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["alert"] is None


class TestFeedbackEndpoint:

    @pytest.mark.asyncio
    async def test_confirm_feedback(self, async_client: httpx.AsyncClient):
        cam = f"CAM_TEST_API_CONFIRM_{uuid.uuid4().hex[:6]}"
        event_payload = {
            "pipeline": "dog_attack",
            "raw_confidence": 0.88,
            "camera_id": cam,
            "zone_id": "default",
            "zone_label": f"Park Area {uuid.uuid4().hex[:6]}",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        res = await async_client.post("/events", json=event_payload)
        assert res.status_code == 200
        alert = res.json()["alert"]
        assert alert is not None
        alert_id = alert["alert_id"]

        try:
            feedback_payload = {
                "alert_id": alert_id,
                "action": "confirm"
            }
            res_fb = await async_client.post("/feedback", json=feedback_payload)
            assert res_fb.status_code == 200
            fb_data = res_fb.json()
            assert fb_data["status"] == "confirmed"
            assert fb_data["alert_id"] == alert_id
        finally:
            await cleanup_alert_log(alert_id)

    @pytest.mark.asyncio
    async def test_dismiss_feedback(self, async_client: httpx.AsyncClient):
        camera_id = f"CAM_TEST_API_DISMISS_{uuid.uuid4().hex[:6]}"
        event_payload = {
            "pipeline": "accident",
            "raw_confidence": 0.82,
            "camera_id": camera_id,
            "zone_id": "default",
            "zone_label": f"Loading Dock {uuid.uuid4().hex[:6]}",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        res = await async_client.post("/events", json=event_payload)
        assert res.status_code == 200
        alert = res.json()["alert"]
        assert alert is not None
        alert_id = alert["alert_id"]

        try:
            feedback_payload = {
                "alert_id": alert_id,
                "action": "dismiss",
                "permanent": True
            }
            res_fb = await async_client.post("/feedback", json=feedback_payload)
            assert res_fb.status_code == 200
            fb_data = res_fb.json()
            assert fb_data["status"] == "dismissed"
            assert fb_data["suppression_rule_id"] is not None
            assert fb_data["permanent"] is True
        finally:
            await cleanup_alert_log(alert_id)
            await cleanup_suppression_rule_by_camera(camera_id)

    @pytest.mark.asyncio
    async def test_feedback_non_existent_alert_returns_404(self, async_client: httpx.AsyncClient):
        fake_id = "alert_999999999999_acc_CAM_FAKE"
        feedback_payload = {
            "alert_id": fake_id,
            "action": "confirm"
        }
        res = await async_client.post("/feedback", json=feedback_payload)
        assert res.status_code == 404
