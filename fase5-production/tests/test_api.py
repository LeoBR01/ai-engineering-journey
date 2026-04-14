# fase5-production/tests/test_api.py
from unittest.mock import MagicMock, patch

import pytest

import src.api as api_module


@pytest.fixture()
def client():
    """TestClient com todos os singletons mockados."""
    from fastapi.testclient import TestClient

    mock_cache = MagicMock()
    mock_rag = MagicMock()
    mock_agent = MagicMock()
    mock_cache.get.return_value = None
    mock_rag.query.return_value = "RAG answer"
    mock_rag.stream.return_value = iter(["tok1", "tok2"])
    mock_agent.run.return_value = "Agent answer"

    with (
        patch.object(api_module, "_cache", mock_cache),
        patch.object(api_module, "_rag", mock_rag),
        patch.object(api_module, "_agent", mock_agent),
        patch("src.api.maybe_schedule_eval"),
    ):
        yield TestClient(api_module.app, raise_server_exceptions=True)


def test_query_rag_returns_answer(client) -> None:
    with patch("src.api.route", return_value="rag"):
        response = client.post("/query", json={"question": "What is RAG?"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "RAG answer"
    assert data["solver"] == "rag"
    assert data["cached"] is False


def test_query_agent_returns_answer(client) -> None:
    with patch("src.api.route", return_value="agent"):
        response = client.post(
            "/query", json={"question": "Compare RAG vs fine-tuning"}
        )
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Agent answer"
    assert data["solver"] == "agent"


def test_query_returns_cache_hit(client) -> None:
    api_module._cache.get.return_value = "cached answer"
    response = client.post("/query", json={"question": "What is RAG?"})
    assert response.status_code == 200
    data = response.json()
    assert data["cached"] is True
    assert data["solver"] == "cache"
    assert data["answer"] == "cached answer"


def test_query_empty_question_returns_400(client) -> None:
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 400


def test_query_missing_question_returns_400(client) -> None:
    response = client.post("/query", json={})
    assert response.status_code == 400


def test_health_endpoint_structure(client) -> None:
    with (
        patch("src.api._check_ollama", return_value=True),
        patch("src.api._check_chromadb", return_value=True),
    ):
        response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "ollama" in data
    assert "chromadb" in data
    assert data["status"] == "ok"


def test_health_degraded_when_ollama_down(client) -> None:
    with (
        patch("src.api._check_ollama", return_value=False),
        patch("src.api._check_chromadb", return_value=True),
    ):
        response = client.get("/health")
    assert response.json()["status"] == "degraded"
