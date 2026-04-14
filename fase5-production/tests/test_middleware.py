import json
import logging

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from src.middleware import LoggingMiddleware


def _make_app() -> FastAPI:
    """App mínima para testar o middleware."""
    app = FastAPI()
    app.add_middleware(LoggingMiddleware)

    @app.get("/ping")
    async def ping():
        return {"pong": True}

    @app.post("/with-state")
    async def with_state(request: Request):
        request.state.solver = "rag"
        request.state.cache_hit = False
        request.state.question_hash = "abc12345"
        return {"ok": True}

    return app


def test_request_returns_200() -> None:
    client = TestClient(_make_app())
    assert client.get("/ping").status_code == 200


def test_trace_id_is_logged(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="src.middleware"):
        TestClient(_make_app()).get("/ping")
    entries = [r for r in caplog.records if "trace_id" in r.message]
    assert len(entries) == 1
    entry = json.loads(entries[0].message)
    assert len(entry["trace_id"]) == 8


def test_log_has_required_fields(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="src.middleware"):
        TestClient(_make_app()).get("/ping")
    entries = [r for r in caplog.records if "trace_id" in r.message]
    entry = json.loads(entries[0].message)
    for field in ("trace_id", "method", "path", "status_code", "latency_ms"):
        assert field in entry, f"campo '{field}' ausente no log"


def test_optional_fields_logged_when_set(caplog) -> None:
    app = _make_app()
    with caplog.at_level(logging.INFO, logger="src.middleware"):
        TestClient(app).post("/with-state")
    entries = [r for r in caplog.records if "trace_id" in r.message]
    entry = json.loads(entries[0].message)
    assert entry["solver"] == "rag"
    assert entry["cache_hit"] is False
    assert entry["question_hash"] == "abc12345"


def test_optional_fields_absent_for_non_query(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="src.middleware"):
        TestClient(_make_app()).get("/ping")
    entries = [r for r in caplog.records if "trace_id" in r.message]
    entry = json.loads(entries[0].message)
    assert "question_hash" not in entry
    assert "solver" not in entry
