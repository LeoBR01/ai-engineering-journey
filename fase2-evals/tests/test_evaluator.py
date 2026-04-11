"""Testes unitários para o orquestrador de avaliação."""

from unittest.mock import patch

from src.evaluator import EvalEntry, evaluate_dataset

SAMPLE_ENTRIES: list[EvalEntry] = [
    {
        "id": "q001",
        "question": "O que é RAG?",
        "expected_answer": "Retrieval-Augmented Generation",
        "source_document": "paper.pdf",
        "relevant_chunk_ids": ["chunk_1"],
    }
]

MOCK_RETRIEVED = [
    {"id": "chunk_1", "text": "RAG significa Retrieval-Augmented Generation."},
    {"id": "chunk_2", "text": "Outro trecho relevante."},
    {"id": "chunk_3", "text": "Mais um trecho."},
    {"id": "chunk_4", "text": "Trecho 4."},
    {"id": "chunk_5", "text": "Trecho 5."},
]


def mock_retrieve(**kwargs):
    return MOCK_RETRIEVED


def mock_generate(**kwargs):
    return "RAG é Retrieval-Augmented Generation."


@patch("src.evaluator.faithfulness", return_value=0.9)
@patch("src.evaluator.answer_relevance", return_value=0.8)
def test_evaluate_dataset_estrutura(mock_relevance, mock_faith):
    report = evaluate_dataset(
        SAMPLE_ENTRIES,
        retrieve_fn=mock_retrieve,
        generate_fn=mock_generate,
    )

    assert report["total_entries"] == 1
    assert len(report["entries"]) == 1

    entry = report["entries"][0]
    assert entry["id"] == "q001"
    assert entry["generated_answer"] == "RAG é Retrieval-Augmented Generation."
    assert entry["faithfulness"] == 0.9
    assert entry["answer_relevance"] == 0.8


@patch("src.evaluator.faithfulness", return_value=1.0)
@patch("src.evaluator.answer_relevance", return_value=1.0)
def test_evaluate_dataset_metricas_retrieval(mock_relevance, mock_faith):
    """chunk_1 está na posição 0 — Recall@3=1.0, MRR=1.0."""
    report = evaluate_dataset(
        SAMPLE_ENTRIES,
        retrieve_fn=mock_retrieve,
        generate_fn=mock_generate,
    )

    assert report["avg_recall_at_3"] == 1.0
    assert report["avg_recall_at_5"] == 1.0
    assert report["avg_mrr"] == 1.0


@patch("src.evaluator.faithfulness", return_value=0.5)
@patch("src.evaluator.answer_relevance", return_value=0.5)
def test_evaluate_dataset_medias_geracao(mock_relevance, mock_faith):
    report = evaluate_dataset(
        SAMPLE_ENTRIES,
        retrieve_fn=mock_retrieve,
        generate_fn=mock_generate,
    )
    assert report["avg_faithfulness"] == 0.5
    assert report["avg_answer_relevance"] == 0.5


@patch("src.evaluator.faithfulness", return_value=0.0)
@patch("src.evaluator.answer_relevance", return_value=0.0)
def test_evaluate_dataset_vazio(mock_relevance, mock_faith):
    report = evaluate_dataset(
        [],
        retrieve_fn=mock_retrieve,
        generate_fn=mock_generate,
    )
    assert report["total_entries"] == 0
    assert report["avg_recall_at_3"] == 0.0
    assert report["avg_mrr"] == 0.0


@patch("src.evaluator.faithfulness", return_value=1.0)
@patch("src.evaluator.answer_relevance", return_value=1.0)
def test_evaluate_dataset_chunk_nao_encontrado(mock_relevance, mock_faith):
    """Relevant chunk que não aparece no retrieval → Recall=0, MRR=0."""
    entries: list[EvalEntry] = [
        {
            "id": "q002",
            "question": "Pergunta difícil",
            "expected_answer": "Resposta esperada",
            "source_document": "doc.pdf",
            "relevant_chunk_ids": ["chunk_ausente"],
        }
    ]
    report = evaluate_dataset(
        entries,
        retrieve_fn=mock_retrieve,
        generate_fn=mock_generate,
    )
    assert report["avg_recall_at_3"] == 0.0
    assert report["avg_mrr"] == 0.0
