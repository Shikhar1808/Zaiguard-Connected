"""
ZaiGuard Alert Engine — FastAPI Application
=============================================
The HTTP interface to the Alert Engine. This is what teammates'
modules actually call.

ENDPOINTS
---------
POST /events     — upstream classifiers POST a RawDetectionEvent here.
                    Returns the DashboardAlert if one was emitted, or
                    a 204-style {"alert": null} if the event was dropped.

POST /feedback    — dashboard POSTs OperatorFeedback here when an
                    operator dismisses or confirms an alert.
                    (Full implementation in Step 10 — stub for now.)

GET  /health      — pings Postgres, Redis, Qdrant. Used by ops/monitoring
                    to confirm all dependencies are reachable.

GET  /config      — returns current threshold/tier/zone config as JSON.
                    Lets the dashboard display current settings.

PUT  /config/reload — forces an immediate config cache refresh, bypassing
                    the TTL. Called after the dashboard updates a value
                    in Postgres directly, so the change takes effect
                    without waiting up to 60 seconds.

STARTUP / SHUTDOWN
------------------
FastAPI's lifespan context manager replaces the older @app.on_event
startup/shutdown decorators. On startup: load config, ensure Qdrant
collections exist. On shutdown: close all connections cleanly.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import structlog
from fastapi import Depends, FastAPI, HTTPException
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from config.database import (
    async_session_factory,
    close_all_connections,
    get_db_session,
    get_qdrant_client,
    get_redis_client,
    health_check,
)
from config.logging import setup_logging
from config.thresholds import config_loader
from feedback.handler import process_feedback, process_outbox_batch
from layers.suppression.semantic import ensure_qdrant_collections
from models.schemas import OperatorFeedback, RawDetectionEvent
from pipeline import process_event

# Set up structured JSON/Console logger
setup_logging()
logger = structlog.get_logger(__name__)


# ─────────────────────────────────────────────────────────────
# Background outbox worker task
# ─────────────────────────────────────────────────────────────

async def _outbox_worker_loop() -> None:
    """
    Runs continuously while the app is alive, draining the outbox
    table every 5 seconds. Each iteration opens its own DB session
    so failures in one batch don't affect the next.
    """
    logger.info("Outbox worker started.")
    while True:
        try:
            async with async_session_factory() as session:
                processed = await process_outbox_batch(
                    session=session,
                    qdrant_client=get_qdrant_client(),
                )
                if processed:
                    logger.info(
                        "outbox.worker.drained", extra={"processed": processed}
                    )
        except Exception as e:
            logger.error("outbox.worker.error: %s", e, exc_info=True)

        await asyncio.sleep(5)


# ─────────────────────────────────────────────────────────────
# Lifespan — startup and shutdown
# ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Alert Engine starting up...")

    await config_loader.ensure_fresh()
    logger.info("Threshold config loaded.")

    qdrant = get_qdrant_client()
    await ensure_qdrant_collections(qdrant)
    logger.info("Qdrant collections verified.")

    # Start outbox worker as a background asyncio task
    outbox_task = asyncio.create_task(_outbox_worker_loop())
    logger.info("Outbox worker task started.")
    logger.info("Alert Engine ready.")

    yield   # app runs here

    outbox_task.cancel()
    try:
        await outbox_task
    except asyncio.CancelledError:
        pass

    logger.info("Alert Engine shutting down...")
    await close_all_connections()
    logger.info("All connections closed.")


app = FastAPI(
    title="ZaiGuard Alert Engine",
    description="Alert generation, thresholding, and suppression pipeline",
    version="0.1.0",
    lifespan=lifespan,
)

from middleware import StructuredLoggingMiddleware
app.add_middleware(StructuredLoggingMiddleware)


# ─────────────────────────────────────────────────────────────
# Dependency providers
# ─────────────────────────────────────────────────────────────

def get_redis() -> Redis:
    return get_redis_client()


def get_qdrant() -> AsyncQdrantClient:
    return get_qdrant_client()


# ─────────────────────────────────────────────────────────────
# POST /events
# ─────────────────────────────────────────────────────────────

@app.post("/events", response_model=dict)
async def receive_event(
    event: RawDetectionEvent,
    session: AsyncSession = Depends(get_db_session),
    redis_client: Redis = Depends(get_redis),
    qdrant_client: AsyncQdrantClient = Depends(get_qdrant),
):
    """
    Receives a raw detection event and runs it through the full
    five-layer pipeline.

    Returns {"alert": DashboardAlert} if the event survived all
    layers, or {"alert": null} if dropped at any gate. A null
    response is NOT an error — upstream classifiers should not retry.
    """
    alert = await process_event(event, redis_client, session, qdrant_client)
    return {"alert": alert.model_dump() if alert is not None else None}


# ─────────────────────────────────────────────────────────────
# POST /feedback
# ─────────────────────────────────────────────────────────────

@app.post("/feedback")
async def receive_feedback(
    feedback: OperatorFeedback,
    session: AsyncSession = Depends(get_db_session),
    redis_client: Redis = Depends(get_redis),
):
    """
    Processes operator feedback from the dashboard.

    DISMISS: writes a suppression_rules row + outbox row (for Qdrant
    embedding) in a single Postgres transaction. The outbox worker
    running in the background will pick up the embedding write within
    ~5 seconds.

    CONFIRM: marks the alert confirmed in alert_log and clears the
    Redis dedup key so the incident keeps producing fresh alerts.

    The dashboard must include the full alert details in the request
    body so the handler can build the suppression rule without an
    extra DB roundtrip.
    """
    # Retrieve the alert from alert_log — needed to build the rule
    result = await session.execute(
        text("""
            SELECT alert_id, pipeline, camera_id, zone_id, zone_label,
                   raw_confidence, effective_conf, timestamp,
                   evidence_frame_ref, involved_ids, suppression_score, tier
            FROM alert_log
            WHERE alert_id = :alert_id
        """),
        {"alert_id": feedback.alert_id},
    )
    row = result.fetchone()

    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Alert {feedback.alert_id} not found in alert_log",
        )

    # Reconstruct a minimal DashboardAlert from the row
    from datetime import datetime, timezone
    from models.schemas import AlertTier, DashboardAlert, Pipeline

    alert = DashboardAlert(
        alert_id=row[0],
        pipeline=Pipeline(row[1]),
        camera_id=row[2],
        zone_id=row[3],
        zone_label=row[4],
        raw_confidence=row[5],
        effective_conf=row[6],
        timestamp=row[7] if row[7].tzinfo else row[7].replace(tzinfo=timezone.utc),
        evidence_frame_ref=row[8],
        involved_ids=row[9] or [],
        suppression_score=row[10] or 0.0,
        tier=AlertTier(row[11]),
    )

    result = await process_feedback(
        feedback=feedback,
        alert=alert,
        session=session,
        redis_client=redis_client,
    )
    return result


# ─────────────────────────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """
    Pings Postgres, Redis, Qdrant.
    Returns 200 if all healthy, 503 if any dependency is unreachable.
    """
    status = await health_check()
    if not all(status.values()):
        raise HTTPException(status_code=503, detail=status)
    return {"status": "healthy", "services": status}


# ─────────────────────────────────────────────────────────────
# GET /config
# ─────────────────────────────────────────────────────────────

@app.get("/config")
async def get_config():
    """Returns the currently cached threshold/tier/zone config."""
    await config_loader.ensure_fresh()
    cache = config_loader._cache
    return {
        "thresholds": {
            pipeline: {
                "base_threshold":   values[0],
                "dedup_ttl_seconds": values[1],
                "escalation_delta": values[2],
            }
            for pipeline, values in cache.thresholds.items()
        },
        "time_multipliers": [
            {"hour_start": h0, "hour_end": h1, "multiplier": m}
            for h0, h1, m in cache.time_multipliers
        ],
        "zone_multipliers":      cache.zone_multipliers,
        "tier_rules":            cache.tier_rules,
        "similarity_thresholds": cache.similarity_thresholds,
    }


# ─────────────────────────────────────────────────────────────
# PUT /config/reload
# ─────────────────────────────────────────────────────────────

@app.put("/config/reload")
async def reload_config():
    """Forces an immediate config reload from Postgres."""
    await config_loader.force_reload()
    return {"status": "reloaded"}


# ─────────────────────────────────────────────────────────────
# Root
# ─────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"service": "ZaiGuard Alert Engine", "version": "0.1.0", "docs": "/docs"}