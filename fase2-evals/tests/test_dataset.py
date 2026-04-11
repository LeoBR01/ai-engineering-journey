"""Testes unitários para o módulo dataset."""

import json
from pathlib import Path

import pytest

from src.dataset import EvalEntry, load_dataset, save_dataset, validate_dataset


@pytest.fixture
def sample_entries() -> list[EvalEntry]:
    return [
        {
            "id": "q001",
            "question": "O que é RAG?",
            "expected_answer": "Retrieval-Augmented Generation",
            "source_document": "paper.pdf",
            "relevant_chunk_ids": ["chunk_1", "chunk_2"],
        },
        {
            "id": "q002",
            "question": "Qual é o objetivo do fine-tuning?",
            "expected_answer": "Adaptar um modelo pré-treinado a uma tarefa específica.",  # noqa: E501
            "source_document": "outro_paper.pdf",
            "relevant_chunk_ids": [],
        },
    ]


def test_save_and_load_dataset(tmp_path: Path, sample_entries: list[EvalEntry]):
    path = tmp_path / "test_dataset.json"
    save_dataset(sample_entries, path)
    loaded = load_dataset(path)
    assert len(loaded) == 2
    assert loaded[0]["id"] == "q001"
    expected = "Adaptar um modelo pré-treinado a uma tarefa específica."
    assert loaded[1]["expected_answer"] == expected


def test_load_dataset_file_not_found(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "inexistente.json")


def test_load_dataset_not_a_list(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"key": "value"}))
    with pytest.raises(ValueError, match="lista JSON"):
        load_dataset(path)


def test_load_dataset_missing_field(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([{"id": "q001", "question": "x"}]))
    with pytest.raises(ValueError, match="faltando campos"):
        load_dataset(path)


def test_validate_dataset_no_warnings(sample_entries: list[EvalEntry]):
    warnings = validate_dataset(sample_entries)
    assert warnings == []


def test_validate_dataset_empty_question(sample_entries: list[EvalEntry]):
    sample_entries[0]["question"] = "  "
    warnings = validate_dataset(sample_entries)
    assert any("question" in w for w in warnings)


def test_validate_dataset_duplicate_id(sample_entries: list[EvalEntry]):
    sample_entries[1]["id"] = "q001"
    warnings = validate_dataset(sample_entries)
    assert any("duplicado" in w for w in warnings)


def test_load_dataset_optional_chunk_ids(tmp_path: Path):
    """Entradas sem relevant_chunk_ids devem receber lista vazia por padrão."""
    data = [
        {
            "id": "q001",
            "question": "Pergunta",
            "expected_answer": "Resposta",
            "source_document": "doc.pdf",
        }
    ]
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(data))
    loaded = load_dataset(path)
    assert loaded[0]["relevant_chunk_ids"] == []
