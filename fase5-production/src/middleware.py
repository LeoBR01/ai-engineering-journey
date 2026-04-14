"""Middleware de logging estruturado JSON com trace_id por request.

Injeta um trace_id único (8 chars hex) em cada request via request.state.
Loga ao final de cada request: trace_id, method, path, status_code, latency_ms.
Campos opcionais (question_hash, solver, cache_hit) são logados se setados pelo handler.
"""

import json
import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Injeta trace_id e loga métricas de cada request em formato JSON."""

    async def dispatch(self, request: Request, call_next) -> Response:
        trace_id = uuid.uuid4().hex[:8]
        request.state.trace_id = trace_id
        start = time.perf_counter()

        response = await call_next(request)

        latency_ms = int((time.perf_counter() - start) * 1000)

        entry: dict = {
            "trace_id": trace_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "latency_ms": latency_ms,
        }

        # Campos opcionais setados pelo handler de /query
        for field in ("question_hash", "solver", "cache_hit"):
            value = getattr(request.state, field, None)
            if value is not None:
                entry[field] = value

        logger.info(json.dumps(entry))
        return response
