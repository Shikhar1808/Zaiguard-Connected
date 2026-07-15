"""
ZaiGuard Alert Engine — Connection Management
================================================
Single place where every connection to Postgres, Redis, and Qdrant
is created. No other file in the codebase should construct these
clients directly — always import from here.

WHY THIS MATTERS FOR INTEGRATION
----------------------------------
If a teammate needs to query the alert log or check Redis state,
they import get_db_session / get_redis_client / get_qdrant_client
from this one file. Connection pooling, retries, and lifecycle are
handled once, here, instead of being reinvented in every layer.

ASYNC THROUGHOUT
-----------------
FastAPI's main advantage is handling many concurrent requests
without blocking threads. That only works if every I/O call in the
request path is async. Postgres uses asyncpg, Redis uses the
built-in asyncio client, Qdrant's client offers an async interface.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import redis.asyncio as redis
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from config.settings import settings

# ─────────────────────────────────────────────────────────────
# PostgreSQL / TimescaleDB — async SQLAlchemy engine
# ─────────────────────────────────────────────────────────────

# pool_size / max_overflow are conservative defaults for a single
# campus deployment. Tune upward only if profiling shows the pool
# is the bottleneck — premature pool tuning is a common time-sink.
engine: AsyncEngine = create_async_engine(
    settings.database_url,
    echo=(settings.log_level == "DEBUG" and not settings.is_production),
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,   # detects stale connections before using them
)

# async_sessionmaker produces new AsyncSession objects on demand.
# expire_on_commit=False means objects remain usable after commit
# without triggering a fresh DB round-trip — important since several
# layers read a value, commit, then immediately use that same value.
async_session_factory = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
    autoflush=False,
)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency — yields a session, guarantees it's closed
    afterward even if the request raises an exception.

    Usage in a route:
        @router.post("/events")
        async def receive_event(
            event: RawDetectionEvent,
            session: AsyncSession = Depends(get_db_session),
        ):
            ...
    """
    async with async_session_factory() as session:
        try:
            yield session
        finally:
            await session.close()


@asynccontextmanager
async def db_session_scope() -> AsyncGenerator[AsyncSession, None]:
    """
    Context-manager version for use OUTSIDE FastAPI routes —
    background workers, scripts, tests.

    Usage:
        async with db_session_scope() as session:
            result = await session.execute(...)
    """
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ─────────────────────────────────────────────────────────────
# Redis — burst dedup (Layer 3), running confidence values
# ─────────────────────────────────────────────────────────────

# A single shared connection pool — redis-py manages pooling
# internally, so one client instance is reused everywhere rather
# than opening a new connection per request.
_redis_client: redis.Redis | None = None


def get_redis_client() -> redis.Redis:
    """
    Returns a shared async Redis client.
    Lazily constructed on first call, then reused (singleton pattern —
    same reasoning as the Postgres engine being created once at
    module import time).
    """
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            decode_responses=True,   # get back str, not bytes
        )
    return _redis_client


# ─────────────────────────────────────────────────────────────
# Qdrant — semantic suppression (Layer 4B)
# ─────────────────────────────────────────────────────────────

_qdrant_client: AsyncQdrantClient | None = None


def get_qdrant_client() -> AsyncQdrantClient:
    """
    Returns a shared async Qdrant client.
    Collection creation/existence is handled separately in
    scripts/init_qdrant.py (run once before first use) — this
    function only returns the connection, it does not assume
    collections already exist.
    """
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = AsyncQdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
    return _qdrant_client


# ─────────────────────────────────────────────────────────────
# Lifecycle — called from main.py's FastAPI startup/shutdown
# ─────────────────────────────────────────────────────────────

async def close_all_connections() -> None:
    """
    Call this on FastAPI shutdown to release connections cleanly.
    Wired into main.py's lifespan handler in Step 9.
    """
    global _redis_client, _qdrant_client

    await engine.dispose()

    if _redis_client is not None:
        await _redis_client.aclose()
        _redis_client = None

    if _qdrant_client is not None:
        await _qdrant_client.close()
        _qdrant_client = None


async def health_check() -> dict[str, bool]:
    """
    Pings all three services. Used by the GET /health endpoint (Step 9)
    so the dashboard/ops team can see at a glance whether the Alert
    Engine's dependencies are reachable.
    """
    status = {"postgres": False, "redis": False, "qdrant": False}

    try:
        async with db_session_scope() as session:
            from sqlalchemy import text
            await session.execute(text("SELECT 1"))
        status["postgres"] = True
    except Exception:
        pass

    try:
        client = get_redis_client()
        await client.ping()
        status["redis"] = True
    except Exception:
        pass

    try:
        client = get_qdrant_client()
        await client.get_collections()
        status["qdrant"] = True
    except Exception:
        pass

    return status