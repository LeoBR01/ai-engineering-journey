"""
Gerenciamento do dataset de avaliação.

Cada entrada do dataset contém uma pergunta, a resposta esperada,
palavras-chave esperadas nos chunks relevantes, nível de dificuldade
e (opcionalmente) os IDs dos chunks relevantes para métricas de retrieval.
"""

import json
from pathlib import Path
from typing import NotRequired, TypedDict


class EvalEntry(TypedDict):
    """
    Uma entrada do dataset de avaliação.

    Campos obrigatórios: id, question, expected_answer.
    Campos opcionais: keywords, difficulty, source_document, relevant_chunk_ids.
    """

    id: str
    question: str
    expected_answer: str
    keywords: NotRequired[list[str]]
    difficulty: NotRequired[str]
    source_document: NotRequired[str]
    relevant_chunk_ids: NotRequired[list[str]]


# Caminho padrão do dataset
DEFAULT_DATASET_PATH = Path(__file__).parent.parent / "data" / "eval_dataset.json"


def load_dataset(path: Path = DEFAULT_DATASET_PATH) -> list[EvalEntry]:
    """
    Carrega o dataset de avaliação a partir de um arquivo JSON.

    Args:
        path: Caminho para o arquivo JSON do dataset.

    Returns:
        Lista de entradas de avaliação.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
        ValueError: Se o JSON estiver malformado ou faltar campos obrigatórios.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("O dataset deve ser uma lista JSON de objetos.")

    entries: list[EvalEntry] = []
    required_fields = {"id", "question", "expected_answer"}

    for i, item in enumerate(data):
        missing = required_fields - item.keys()
        if missing:
            raise ValueError(f"Entrada {i} faltando campos: {missing}")
        entry = EvalEntry(
            id=item["id"],
            question=item["question"],
            expected_answer=item["expected_answer"],
        )
        if "keywords" in item:
            entry["keywords"] = item["keywords"]
        if "difficulty" in item:
            entry["difficulty"] = item["difficulty"]
        if "source_document" in item:
            entry["source_document"] = item["source_document"]
        entry["relevant_chunk_ids"] = item.get("relevant_chunk_ids", [])
        entries.append(entry)

    return entries


def save_dataset(entries: list[EvalEntry], path: Path = DEFAULT_DATASET_PATH) -> None:
    """
    Persiste o dataset de avaliação em um arquivo JSON.

    Args:
        entries: Lista de entradas a salvar.
        path: Caminho de destino do arquivo JSON.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


def validate_dataset(entries: list[EvalEntry]) -> list[str]:
    """
    Valida a consistência do dataset e retorna lista de avisos.

    Args:
        entries: Lista de entradas a validar.

    Returns:
        Lista de strings com avisos (vazia se tudo estiver correto).
    """
    warnings: list[str] = []
    ids_vistos: set[str] = set()

    for entry in entries:
        if not entry["question"].strip():
            warnings.append(f"[{entry['id']}] question está vazia.")
        if not entry["expected_answer"].strip():
            warnings.append(f"[{entry['id']}] expected_answer está vazia.")
        if entry["id"] in ids_vistos:
            warnings.append(f"[{entry['id']}] ID duplicado no dataset.")
        ids_vistos.add(entry["id"])

    return warnings
