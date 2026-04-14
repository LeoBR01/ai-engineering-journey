# fase5-production/tests/test_integration.py
"""Testes de integração — requerem Ollama rodando e ChromaDB da Fase 1 populado.

Executar com:
    uv run pytest tests/test_integration.py -v -m integration
"""

import pytest
from fastapi.testclient import TestClient

from src.api import app

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_check_all_services_ok(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["ollama"] is True, "Ollama não está rodando — execute: ollama serve"
    assert data["chromadb"] is True


def test_query_rag_returns_non_empty_answer(client) -> None:
    response = client.post("/query", json={"question": "What is RAG?"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["answer"]) > 20
    assert data["solver"] in ("rag", "cache")
    assert isinstance(data["cached"], bool)


def test_query_agent_routes_for_comparison(client) -> None:
    response = client.post(
        "/query",
        json={"question": "Compare RAG and fine-tuning approaches"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["solver"] in ("agent", "cache")


def test_second_query_may_hit_cache(client) -> None:
    question = "What is attention mechanism in transformers?"
    # Primeira chamada — popula cache
    r1 = client.post("/query", json={"question": question})
    assert r1.status_code == 200

    # Segunda chamada — pode ou não vir do cache dependendo do threshold
    r2 = client.post("/query", json={"question": question})
    assert r2.status_code == 200
    # Se vier do cache, cached deve ser True
    if r2.json()["solver"] == "cache":
        assert r2.json()["cached"] is True
