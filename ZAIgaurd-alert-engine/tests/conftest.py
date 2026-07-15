"""
pytest configuration — test data hygiene.

EVENT LOOP
----------
No custom event_loop fixture here anymore. Modern pytest-asyncio
(>=0.24) handles session-wide loop sharing natively via two ini
settings in pytest.ini:

    asyncio_default_fixture_loop_scope = session
    asyncio_default_test_loop_scope = session

These make every async test AND every async fixture in the session
share exactly one event loop. This is what fixes the Windows
"Future attached to a different loop" errors that occurred when a
custom event_loop fixture was only partially honored by some fixtures.

TEST DATA HYGIENE
-----------------
Cleans CAM_TEST_* suppression rules before/after the session so
stale rows from crashed previous runs never contaminate future tests.
Soft-fails silently if Postgres is unreachable -- pure-logic tests
(schemas, thresholds, dedup, tiering) don't need a database and must
still run cleanly without it.
"""

import asyncio
import pytest
import pytest_asyncio
from sqlalchemy import text



@pytest_asyncio.fixture(scope="session", autouse=True)
async def cleanup_test_rules():
    """
    Deletes CAM_TEST_* suppression rules before and after the session,
    and clears Qdrant dismissed_alerts so semantic suppression history
    from prior test runs does not affect new runs.
    Silently skips if Postgres or Qdrant is unreachable.
    """
    from config.database import db_session_scope
    from config.settings import settings
    from qdrant_client import AsyncQdrantClient

    async def _delete():
        try:
            async with db_session_scope() as session:
                await session.execute(
                    text(
                        "DELETE FROM suppression_rules "
                        "WHERE camera_id LIKE 'CAM_%'"
                    )
                )
        except Exception:
            pass

        try:
            qclient = AsyncQdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
            await qclient.delete_collection(settings.qdrant_collection_dismissed)
            await qclient.close()
        except Exception:
            pass

    await _delete()
    yield
    await _delete()