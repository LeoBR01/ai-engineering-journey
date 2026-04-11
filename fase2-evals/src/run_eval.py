"""
Script de integração: conecta fase2-evals ao pipeline real da Fase 1.

Requer Ollama rodando com llama3.2 e nomic-embed-text, e o ChromaDB
populado da Fase 1 (~4.110 chunks de papers do ArXiv).

Uso:
    uv run eval-run
    # ou diretamente:
    uv run python -m src.run_eval
"""

from pathlib import Path

import chromadb
import ollama
from rich.console import Console
from rich.table import Table

from .dataset import DEFAULT_DATASET_PATH, load_dataset
from .evaluator import evaluate_dataset
from .report import save_report

# ── Caminhos ──────────────────────────────────────────────────────────────────

FASE1_ROOT = Path(__file__).parent.parent.parent / "fase1-rag"
CHROMA_PATH = str(FASE1_ROOT / "chroma_db")
COLLECTION_NAME = "papers"
GENERATION_MODEL = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"

# ── Helpers ──────────────────────────────────────────────────────────────────


def generate_embedding(text: str, model: str = EMBEDDING_MODEL) -> list[float]:
    """
    Gera o embedding de um texto usando Ollama diretamente.

    Não importa da Fase 1 — chama `ollama.embed` diretamente para evitar
    conflito de pacotes (ambos os projetos têm um pacote `src`).
    Esta função é ponto de patch explícito nos testes unitários.

    Args:
        text: texto a ser convertido em embedding.
        model: nome do modelo de embedding no Ollama.

    Returns:
        Vetor de floats representando o texto.
    """
    response = ollama.embed(model=model, input=text)
    return response.embeddings[0]


# ── Funções de integração ─────────────────────────────────────────────────────


def retrieve_fn(
    query: str,
    top_k: int = 5,
    collection_name: str = COLLECTION_NAME,
    chroma_path: str = CHROMA_PATH,
) -> list[dict]:
    """
    Consulta o ChromaDB da Fase 1 e retorna os chunks mais relevantes.

    Gera o embedding da query com nomic-embed-text e executa busca por
    similaridade cosine, incluindo o ID do chunk no resultado para
    que as métricas de retrieval possam comparar com ground truth.

    Args:
        query: pergunta do usuário.
        top_k: número de chunks a retornar.
        collection_name: nome da collection no ChromaDB.
        chroma_path: caminho para o diretório do ChromaDB.

    Returns:
        Lista de dicts com 'id', 'text', 'source', 'page' e 'score'.
    """
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_collection(name=collection_name)

    n_results = min(top_k, collection.count())
    if n_results == 0:
        return []

    embedding = generate_embedding(query, model=EMBEDDING_MODEL)

    results = collection.query(
        query_embeddings=[embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    return [
        {
            "id": chunk_id,
            "text": doc,
            "source": meta.get("source", ""),
            "page": meta.get("page", 0),
            "score": round(1 - dist, 4),
        }
        for chunk_id, doc, meta, dist in zip(ids, docs, metas, distances, strict=True)
    ]


def generate_fn(question: str, context: str) -> str:
    """
    Gera uma resposta usando llama3.2 via Ollama com o contexto fornecido.

    Usa padrão de prompt RAG: instrução de sistema com o contexto
    recuperado + pergunta do usuário.

    Args:
        question: pergunta do usuário.
        context: texto dos chunks recuperados, concatenados.

    Returns:
        Resposta gerada pelo LLM como string.
    """
    system = (
        "You are an assistant that answers questions exclusively based on the "
        "provided context. If the answer is not in the context, say so clearly.\n\n"
        f"Context:\n{context}"
    )
    response = ollama.chat(
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
    )
    return response.message.content


def judge_fn(prompt: str) -> str:
    """
    Chama o llama3.2 com um prompt arbitrário e retorna a resposta raw.

    Primitiva de baixo nível para avaliação LLM-as-judge.
    As funções faithfulness() e answer_relevance() chamam o Ollama
    diretamente; esta função permite invocar o juiz de forma isolada
    em experimentos ad-hoc ou para inspecionar prompts de avaliação.

    Args:
        prompt: texto completo do prompt para o juiz.

    Returns:
        Resposta raw do modelo como string.
    """
    response = ollama.chat(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.message.content


# ── Formatação da tabela ──────────────────────────────────────────────────────


def _score_color(value: float) -> str:
    """Retorna cor rich baseada no score: verde ≥ 0.7, amarelo ≥ 0.4, vermelho < 0.4."""
    if value >= 0.7:
        return "green"
    if value >= 0.4:
        return "yellow"
    return "red"


def _fmt(value: float) -> str:
    """Formata score com cor para a tabela rich."""
    color = _score_color(value)
    return f"[{color}]{value:.2f}[/{color}]"


def print_results_table(report: dict, entries: list[dict]) -> None:
    """
    Exibe tabela rich com resultados por pergunta e linha de médias.

    Colunas: ID, Difficulty, Recall@5, MRR, Faithfulness, Answer Relevance.
    Cores: verde ≥ 0.7, amarelo ≥ 0.4, vermelho < 0.4.

    Args:
        report: EvalReport retornado por evaluate_dataset().
        entries: entradas originais do dataset (com 'difficulty').
    """
    console = Console()

    table = Table(
        title="Fase 2 — Resultados da Avaliação RAG",
        show_lines=True,
    )
    table.add_column("ID", style="bold", width=6)
    table.add_column("Difficulty", width=10)
    table.add_column("Recall@5", justify="center", width=10)
    table.add_column("MRR", justify="center", width=8)
    table.add_column("Faithfulness", justify="center", width=13)
    table.add_column("Ans.Relevance", justify="center", width=14)

    entry_meta = {e["id"]: e for e in entries}

    for result in report["entries"]:
        difficulty = entry_meta.get(result["id"], {}).get("difficulty", "-")
        diff_color = {"easy": "green", "medium": "yellow", "hard": "red"}.get(
            difficulty, "white"
        )
        table.add_row(
            result["id"],
            f"[{diff_color}]{difficulty}[/{diff_color}]",
            _fmt(result["recall_at_5"]),
            _fmt(result["mrr"]),
            _fmt(result["faithfulness"]),
            _fmt(result["answer_relevance"]),
        )

    # Linha de médias
    table.add_row(
        "[bold]AVG[/bold]",
        "—",
        _fmt(report["avg_recall_at_5"]),
        _fmt(report["avg_mrr"]),
        _fmt(report["avg_faithfulness"]),
        _fmt(report["avg_answer_relevance"]),
        style="bold",
    )

    console.print(table)
    console.print(
        "\n[dim]Nota: Recall@5 e MRR são 0.0 porque relevant_chunk_ids não foram "
        "anotados. Faithfulness e Answer Relevance são as métricas principais.[/dim]"
    )


# ── Entrypoint ────────────────────────────────────────────────────────────────


def main() -> None:
    """
    Executa a avaliação completa do pipeline RAG e exibe o relatório.

    Fluxo:
    1. Carrega o dataset de avaliação
    2. Executa retrieval + geração + métricas para cada pergunta
    3. Salva resultados em data/results/ com timestamp
    4. Exibe tabela rich no terminal
    """
    console = Console()
    console.print("\n[bold blue]Fase 2 — Evals: iniciando avaliação...[/bold blue]\n")

    entries = load_dataset(DEFAULT_DATASET_PATH)
    console.print(f"Dataset carregado: [bold]{len(entries)}[/bold] perguntas\n")

    console.print("[dim]Rodando retrieval + geração + LLM-as-judge...[/dim]")
    report = evaluate_dataset(
        entries,
        retrieve_fn=retrieve_fn,
        generate_fn=generate_fn,
    )

    result_path = save_report(report, label="fase1")
    console.print(f"\nResultados salvos em: [green]{result_path}[/green]\n")

    print_results_table(report, entries)


if __name__ == "__main__":
    main()
