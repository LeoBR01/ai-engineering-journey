# tests/test_cache.py
import time
from unittest.mock import MagicMock, patch

from src.cache import SemanticCache


def _make_cache(threshold: float = 0.92, ttl: int = 86400) -> SemanticCache:
    """Cria SemanticCache sem conectar ao ChromaDB ou Ollama."""
    cache = SemanticCache.__new__(SemanticCache)
    cache._threshold = threshold
    cache._ttl = ttl
    cache._collection = MagicMock()
    return cache


def test_get_miss_no_results() -> None:
    cache = _make_cache()
    cache._collection.query.return_value = {
        "distances": [[]],
        "documents": [[]],
        "metadatas": [[]],
    }
    with patch.object(cache, "_embed", return_value=[0.1] * 768):
        assert cache.get("What is RAG?") is None


def test_get_hit_above_threshold() -> None:
    cache = _make_cache()
    cache._collection.query.return_value = {
        "distances": [[0.04]],  # similarity = 0.96 > 0.92
        "documents": [["cached answer"]],
        "metadatas": [[{"question": "q", "cached_at": time.time()}]],
    }
    with patch.object(cache, "_embed", return_value=[0.1] * 768):
        assert cache.get("What is RAG?") == "cached answer"


def test_get_miss_below_threshold() -> None:
    cache = _make_cache()
    cache._collection.query.return_value = {
        "distances": [[0.12]],  # similarity = 0.88 < 0.92
        "documents": [["cached answer"]],
        "metadatas": [[{"question": "q", "cached_at": time.time()}]],
    }
    with patch.object(cache, "_embed", return_value=[0.1] * 768):
        assert cache.get("What is RAG?") is None


def test_get_miss_expired() -> None:
    cache = _make_cache(ttl=3600)
    expired = time.time() - 7200  # 2h atrás — TTL é 1h
    cache._collection.query.return_value = {
        "distances": [[0.04]],
        "documents": [["old answer"]],
        "metadatas": [[{"question": "q", "cached_at": expired}]],
    }
    with patch.object(cache, "_embed", return_value=[0.1] * 768):
        assert cache.get("What is RAG?") is None


def test_set_stores_answer_in_document() -> None:
    cache = _make_cache()
    with patch.object(cache, "_embed", return_value=[0.1] * 768):
        cache.set("question", "my answer")
    call_kwargs = cache._collection.upsert.call_args.kwargs
    assert call_kwargs["documents"] == ["my answer"]
    assert "cached_at" in call_kwargs["metadatas"][0]


def test_set_silently_ignores_errors() -> None:
    cache = _make_cache()
    cache._collection.upsert.side_effect = Exception("chroma error")
    with patch.object(cache, "_embed", return_value=[0.1] * 768):
        cache.set("question", "answer")  # não deve levantar exceção


def test_get_returns_none_on_exception() -> None:
    cache = _make_cache()
    cache._collection.query.side_effect = Exception("chroma error")
    with patch.object(cache, "_embed", return_value=[0.1] * 768):
        assert cache.get("question") is None
