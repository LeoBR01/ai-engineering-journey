"""
Orquestrador da avaliação completa do pipeline RAG.

Itera sobre o dataset de avaliação, executa retrieval e geração para cada
pergunta, aplica as métricas e retorna um relatório estruturado de resultados.

O módulo aceita as funções `retrieve_fn` e `generate_fn` como parâmetros para
permitir injeção de dependência nos testes sem precisar importar a Fase 1.
"""

import sys
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict

from .dataset import EvalEntry
from .metrics_generation import answer_relevance, faithfulness
from .metrics_retrieval import recall_at_k, reciprocal_rank

# Caminho raiz da Fase 1 (pai de src/) — necessário para `from src.X import`
FASE1_ROOT = Path(__file__).parent.parent.parent / "fase1-rag"


def _load_fase1_functions() -> tuple[Callable, Callable]:
    """
    Importa as funções `retrieve` e `generate` da Fase 1 em tempo de execução.

    Faz a importação de forma lazy para que os testes possam injetar mocks
    sem depender do ambiente da Fase 1 (ChromaDB, Ollama).

    Returns:
        Tupla (retrieve, generate) dos módulos da Fase 1.
    """
    fase1_root_str = str(FASE1_ROOT)
    if fase1_root_str not in sys.path:
        sys.path.insert(0, fase1_root_str)

    from src.generation import generate  # noqa: PLC0415
    from src.retrieval import retrieve  # noqa: PLC0415

    return retrieve, generate


class EntryResult(TypedDict):
    """Resultado da avaliação de uma única entrada do dataset."""

    id: str
    question: str
    expected_answer: str
    generated_answer: str
    retrieved_chunk_ids: list[str]
    recall_at_3: float
    recall_at_5: float
    mrr: float
    faithfulness: float
    answer_relevance: float


class EvalReport(TypedDict):
    """Relatório agregado de toda a avaliação."""

    total_entries: int
    avg_recall_at_3: float
    avg_recall_at_5: float
    avg_mrr: float
    avg_faithfulness: float
    avg_answer_relevance: float
    entries: list[EntryResult]


def evaluate_dataset(
    entries: list[EvalEntry],
    top_k: int = 5,
    collection_name: str = "papers",
    chroma_path: str | None = None,
    retrieve_fn: Callable | None = None,
    generate_fn: Callable | None = None,
) -> EvalReport:
    """
    Avalia o pipeline RAG sobre o dataset de avaliação completo.

    Para cada entrada:
    1. Executa retrieval com top_k chunks
    2. Gera resposta com o contexto recuperado
    3. Calcula métricas de retrieval e geração

    Args:
        entries: Lista de entradas do dataset de avaliação.
        top_k: Número de chunks a recuperar por query.
        collection_name: Nome da collection no ChromaDB.
        chroma_path: Caminho para o ChromaDB (padrão: fase1-rag/chroma_db).
        retrieve_fn: Função de retrieval (injeção para testes).
        generate_fn: Função de geração (injeção para testes).

    Returns:
        EvalReport com métricas agregadas e resultados por entrada.
    """
    if chroma_path is None:
        chroma_path = str(FASE1_ROOT / "chroma_db")

    if retrieve_fn is None or generate_fn is None:
        _retrieve, _generate = _load_fase1_functions()
        retrieve_fn = retrieve_fn or _retrieve
        generate_fn = generate_fn or _generate

    entry_results: list[EntryResult] = []

    for entry in entries:
        # Retrieval: busca os chunks mais relevantes
        retrieved = retrieve_fn(
            query=entry["question"],
            top_k=top_k,
            collection_name=collection_name,
            chroma_path=chroma_path,
        )
        retrieved_ids = [chunk["id"] for chunk in retrieved]
        context = "\n\n".join(chunk["text"] for chunk in retrieved)

        # Geração: produz resposta com o contexto recuperado
        generated_answer = generate_fn(
            question=entry["question"],
            context=context,
        )

        # Métricas de retrieval
        relevant_ids = entry["relevant_chunk_ids"]
        r_at_3 = recall_at_k(retrieved_ids, relevant_ids, k=3)
        r_at_5 = recall_at_k(retrieved_ids, relevant_ids, k=5)
        rr = reciprocal_rank(retrieved_ids, relevant_ids)

        # Métricas de geração (LLM-as-judge)
        faith = faithfulness(answer=generated_answer, context=context)
        relevance = answer_relevance(
            answer=generated_answer, question=entry["question"]
        )

        entry_results.append(
            EntryResult(
                id=entry["id"],
                question=entry["question"],
                expected_answer=entry["expected_answer"],
                generated_answer=generated_answer,
                retrieved_chunk_ids=retrieved_ids,
                recall_at_3=r_at_3,
                recall_at_5=r_at_5,
                mrr=rr,
                faithfulness=faith,
                answer_relevance=relevance,
            )
        )

    def avg(key: str) -> float:
        if not entry_results:
            return 0.0
        return sum(r[key] for r in entry_results) / len(entry_results)  # type: ignore[literal-required]

    return EvalReport(
        total_entries=len(entry_results),
        avg_recall_at_3=avg("recall_at_3"),
        avg_recall_at_5=avg("recall_at_5"),
        avg_mrr=avg("mrr"),
        avg_faithfulness=avg("faithfulness"),
        avg_answer_relevance=avg("answer_relevance"),
        entries=entry_results,
    )
