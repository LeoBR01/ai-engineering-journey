# fase5-production/tests/test_rag_adapter.py
from unittest.mock import MagicMock, patch

from src.rag_adapter import RagAdapter


def _make_adapter() -> RagAdapter:
    """Cria RagAdapter sem conectar ao ChromaDB."""
    adapter = RagAdapter.__new__(RagAdapter)
    adapter._collection = MagicMock()
    return adapter


def test_query_calls_retrieve_and_generate() -> None:
    adapter = _make_adapter()
    fake_chunks = [{"text": "RAG content", "source": "paper.pdf", "page": 1}]
    with (
        patch("src.rag_adapter.retrieve", return_value=fake_chunks),
        patch("src.rag_adapter.generate", return_value="RAG answer"),
    ):
        result = adapter.query("What is RAG?")
    assert result == "RAG answer"


def test_query_passes_collection_to_retrieve() -> None:
    adapter = _make_adapter()
    with (
        patch("src.rag_adapter.retrieve", return_value=[]) as mock_retrieve,
        patch("src.rag_adapter.generate", return_value="answer"),
    ):
        adapter.query("question")
    mock_retrieve.assert_called_once_with("question", collection=adapter._collection)


def test_stream_yields_tokens() -> None:
    adapter = _make_adapter()
    fake_chunks = [{"text": "content", "source": "p.pdf", "page": 1}]
    fake_stream = [
        MagicMock(message=MagicMock(content="token1")),
        MagicMock(message=MagicMock(content="token2")),
        MagicMock(message=MagicMock(content="")),  # token vazio — deve ser ignorado
    ]
    with (
        patch("src.rag_adapter.retrieve", return_value=fake_chunks),
        patch("src.rag_adapter.format_context", return_value="formatted context"),
        patch(
            "src.rag_adapter.build_prompt",
            return_value=[{"role": "user", "content": "q"}],
        ),
        patch("src.rag_adapter.ollama.chat", return_value=iter(fake_stream)),
    ):
        tokens = list(adapter.stream("question"))
    assert tokens == ["token1", "token2"]


def test_stream_passes_collection_to_retrieve() -> None:
    adapter = _make_adapter()
    with (
        patch("src.rag_adapter.retrieve", return_value=[]) as mock_retrieve,
        patch("src.rag_adapter.format_context", return_value=""),
        patch("src.rag_adapter.build_prompt", return_value=[]),
        patch("src.rag_adapter.ollama.chat", return_value=iter([])),
    ):
        list(adapter.stream("question"))
    mock_retrieve.assert_called_once_with("question", collection=adapter._collection)
