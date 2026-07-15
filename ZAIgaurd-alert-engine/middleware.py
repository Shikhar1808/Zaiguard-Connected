"""
ZaiGuard Alert Engine — Structured Request Logging Middleware
=============================================================
FastAPI BaseHTTPMiddleware subclass to intercept, trace, and log all
incoming requests and outgoing responses using structlog.
"""

from __future__ import annotations

import time
import uuid
import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

# Use structlog to get a dedicated request logger
logger = structlog.get_logger("api.requests")


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = time.perf_counter()

        # Check for client-provided Request ID, or generate a unique UUID
        request_id = request.headers.get("X-Request-ID", "")
        if not request_id:
            request_id = str(uuid.uuid4())

        # Clear and bind request_id contextvars so that any downstream logs
        # triggered during this request will automatically inherit request_id.
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        response: Response | None = None
        exception: Exception | None = None

        try:
            response = await call_next(request)
            return response
        except Exception as e:
            exception = e
            raise
        finally:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            client_host = request.client.host if request.client else "unknown"

            log_kwargs = {
                "method": request.method,
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "client_host": client_host,
                "duration_ms": duration_ms,
            }

            if exception is not None:
                log_kwargs["status_code"] = 500
                log_kwargs["error"] = str(exception)
                logger.error("request.failed_with_exception", **log_kwargs)
            elif response is not None:
                status_code = response.status_code
                log_kwargs["status_code"] = status_code

                if status_code >= 500:
                    logger.error("request.server_error", **log_kwargs)
                elif status_code >= 400:
                    logger.warning("request.client_error", **log_kwargs)
                else:
                    # Healthy response
                    logger.info("request.success", **log_kwargs)
            else:
                # Edge case: both response and exception are None
                log_kwargs["status_code"] = 500
                logger.error("request.unknown_termination", **log_kwargs)
