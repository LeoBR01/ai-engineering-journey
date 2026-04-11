"""
Testes unitários para run_eval.py — totalmente mockados, sem Ollama ou ChromaDB.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.run_eval import generate_fn, retrieve_fn

# ── Fixtures de resposta mockada ──────────────────────────────────────────────

MOCK_CHROMA_RESULTS = {
    "ids": [["chunk_abc", "chunk_def", "chunk_ghi", "chunk_jkl", "chunk_mno"]],
    "documents": [
        [
            "RAG combines retrieval with generation.",
            "Chunking splits documents into smaller pieces.",
            "Embeddings represent text as vectors.",
            "Vector databases store embeddings for fast search.",
            "Fine-tuning adapts a model to a specific task.",
        ]
    ],
    "metadatas": [
        [
            {"source": "rag_paper.pdf", "page": 1},
            {"source": "rag_paper.pdf", "page": 2},
            {"source": "chunking_paper.pdf", "page": 1},
            {"source": "vector_db.pdf", "page": 3},
            {"source": "fine_tuning.pdf", "page": 5},
        ]
    ],
    "distances": [[0.1, 0.2, 0.3, 0.4, 0.5]],
}


# ── test_retrieve_fn_returns_list ─────────────────────────────────────────────


@patch("src.run_eval.generate_embedding", return_value=[0.1] * 768)
@patch("src.run_eval.chromadb.PersistentClient")
def test_retrieve_fn_returns_list(mock_client_cls, mock_embed):  # noqa: ARG001
    """retrieve_fn deve retornar lista de dicts com campos obrigatórios."""
    mock_collection = MagicMock()
    mock_collection.count.return_value = 100
    mock_collection.query.return_value = MOCK_CHROMA_RESULTS
    mock_client_cls.return_value.get_collection.return_value = mock_collection

    results = retrieve_fn(query="What is RAG?", top_k=5)

    assert isinstance(results, list)
    assert len(results) == 5

    first = results[0]
    assert first["id"] == "chunk_abc"
    assert first["text"] == "RAG combines retrieval with generation."
    assert first["source"] == "rag_paper.pdf"
    assert first["page"] == 1
    assert first["score"] == pytest.approx(0.9)  # 1 - 0.1


@patch("src.run_eval.generate_embedding", return_value=[0.1] * 768)
@patch("src.run_eval.chromadb.PersistentClient")
def test_retrieve_fn_empty_collection(mock_client_cls, mock_embed):
    """retrieve_fn deve retornar lista vazia se a collection estiver vazia."""
    mock_collection = MagicMock()
    mock_collection.count.return_value = 0
    mock_client_cls.return_value.get_collection.return_value = mock_collection

    results = retrieve_fn(query="Any question?", top_k=5)

    assert results == []
    mock_collection.query.assert_not_called()


# ── test_generate_fn_returns_string ──────────────────────────────────────────


@patch("src.run_eval.ollama.chat")
def test_generate_fn_returns_string(mock_chat):
    """generate_fn deve retornar string com a resposta do LLM."""
    mock_response = MagicMock()
    expected = "RAG is a technique that combines retrieval with LLMs."
    mock_response.message.content = expected
    mock_chat.return_value = mock_response

    result = generate_fn(
        question="What is RAG?",
        context="RAG stands for Retrieval-Augmented Generation.",
    )

    assert isinstance(result, str)
    assert len(result) > 0
    assert result == expected


@patch("src.run_eval.ollama.chat")
def test_generate_fn_passes_context_in_system(mock_chat):
    """generate_fn deve incluir o contexto na mensagem de sistema."""
    mock_response = MagicMock()
    mock_response.message.content = "Answer."
    mock_chat.return_value = mock_response

    context = "Important context about RAG."
    generate_fn(question="Question?", context=context)

    call_args = mock_chat.call_args
    messages = call_args[1]["messages"] if call_args[1] else call_args[0][1]
    system_msg = next(m for m in messages if m["role"] == "system")
    assert context in system_msg["content"]


# ── test_full_eval_pipeline_mocked ───────────────────────────────────────────


@patch("src.evaluator.answer_relevance", return_value=0.85)
@patch("src.evaluator.faithfulness", return_value=0.90)
@patch(
    "src.run_eval.generate_fn",
    return_value="RAG combines retrieval with generation.",
)
@patch("src.run_eval.retrieve_fn")
def test_full_eval_pipeline_mocked(
    mock_retrieve, mock_generate, mock_faith, mock_relevance
):
    """
    Executa o pipeline completo de avaliação com todos os componentes mockados.
    Verifica que o relatório tem a estrutura esperada e as métricas corretas.
    """
    from src.dataset import EvalEntry
    from src.evaluator import evaluate_dataset

    mock_retrieve.return_value = [
        {"id": "chunk_1", "text": "RAG combines retrieval with generation."},
        {"id": "chunk_2", "text": "Chunking splits documents."},
        {"id": "chunk_3", "text": "Embeddings are dense vectors."},
        {"id": "chunk_4", "text": "ChromaDB stores vectors."},
        {"id": "chunk_5", "text": "Fine-tuning adapts models."},
    ]

    entries: list[EvalEntry] = [
        {
            "id": "q001",
            "question": "What is RAG?",
            "expected_answer": "Retrieval-Augmented Generation",
            "source_document": "rag_paper.pdf",
            "relevant_chunk_ids": ["chunk_1"],
        }
    ]

    report = evaluate_dataset(
        entries,
        retrieve_fn=mock_retrieve,
        generate_fn=mock_generate,
    )

    assert report["total_entries"] == 1
    assert report["avg_faithfulness"] == pytest.approx(0.90)
    assert report["avg_answer_relevance"] == pytest.approx(0.85)

    # chunk_1 está na posição 0 — Recall@5 = 1.0, MRR = 1.0
    assert report["avg_recall_at_5"] == pytest.approx(1.0)
    assert report["avg_mrr"] == pytest.approx(1.0)

    entry_result = report["entries"][0]
    assert entry_result["generated_answer"] == "RAG combines retrieval with generation."
    assert entry_result["faithfulness"] == pytest.approx(0.90)
    assert entry_result["answer_relevance"] == pytest.approx(0.85)
