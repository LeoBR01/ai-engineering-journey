"""Avaliação comparativa: baseline (Fase 2) vs. modelo fine-tuned.

Reutiliza o sistema de evals da Fase 2 com o modelo parametrizado.

Uso:
    uv run evaluate-model --model llama3.2-finetuned \\
                          --output data/results
"""

import argparse
import json
import logging
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import ollama

logger = logging.getLogger(__name__)

FASE2_ROOT = Path(__file__).resolve().parent.parent.parent / "fase2-evals"
DEFAULT_BASELINE_PATH = str(FASE2_ROOT / "data" / "results")
DEFAULT_DATASET_PATH = str(FASE2_ROOT / "data" / "eval_dataset.json")
EMBEDDING_MODEL = "nomic-embed-text"
FASE1_CHROMA_PATH = str(
    Path(__file__).resolve().parent.parent.parent / "fase1-rag" / "chroma_db"
)


def make_generate_fn(model_name: str) -> Callable[[str, str], str]:
    """Cria uma função generate parametrizada pelo nome do modelo Ollama.

    Permite trocar o modelo de geração sem alterar o evaluator da Fase 2.

    Args:
        model_name: nome do modelo no Ollama (ex: 'llama3.2-finetuned').

    Returns:
        Função compatível com a assinatura esperada pelo evaluate_dataset da Fase 2.
    """

    def generate_fn(question: str, context: str) -> str:
        system = (
            "You are an assistant that answers questions exclusively based on the "
            "provided context. If the answer is not in the context, say so clearly.\n\n"
            f"Context:\n{context}"
        )
        response = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": question},
            ],
        )
        return response.message.content

    return generate_fn


def _load_fase2_functions():
    """Importa evaluate_dataset, load_dataset e save_report da Fase 2."""
    # Adiciona fase2-evals/src/ diretamente para evitar sombreamento do pacote
    # src/ local. Importa com nomes bare (sem prefixo "src.").
    fase2_src = str(FASE2_ROOT / "src")
    if fase2_src not in sys.path:
        sys.path.insert(0, fase2_src)

    from dataset import load_dataset  # noqa: PLC0415
    from evaluator import evaluate_dataset  # noqa: PLC0415
    from report import save_report  # noqa: PLC0415

    return evaluate_dataset, load_dataset, save_report


def _make_retrieve_fn() -> Callable:
    """Cria a função de retrieval usando ChromaDB da Fase 1."""
    import chromadb

    def retrieve_fn(
        query: str,
        top_k: int = 5,
        collection_name: str = "papers",
        chroma_path: str = FASE1_CHROMA_PATH,
    ) -> list[dict]:
        client = chromadb.PersistentClient(path=chroma_path)
        collection = client.get_collection(name=collection_name)

        n_results = min(top_k, collection.count())
        if n_results == 0:
            return []

        embedding_response = ollama.embed(model=EMBEDDING_MODEL, input=query)
        embedding = embedding_response.embeddings[0]

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
            for chunk_id, doc, meta, dist in zip(
                ids, docs, metas, distances, strict=True
            )
        ]

    return retrieve_fn


def _load_latest_baseline(baseline_path: str) -> dict:
    """Carrega o arquivo de resultado mais recente da Fase 2.

    Args:
        baseline_path: diretório com arquivos eval_fase1_*.json.

    Returns:
        Dict com resultados do baseline.

    Raises:
        FileNotFoundError: se nenhum baseline for encontrado.
    """
    baseline_dir = Path(baseline_path)
    json_files = sorted(baseline_dir.glob("eval_fase1_*.json"))

    if not json_files:
        raise FileNotFoundError(
            f"Nenhum baseline encontrado em: {baseline_path}\n"
            "Execute a Fase 2 primeiro: cd ../fase2-evals && uv run eval-run"
        )

    return json.loads(json_files[-1].read_text(encoding="utf-8"))


def _compare_results(
    current: dict,
    baseline: dict,
    model_name: str,
) -> dict:
    """Gera relatório comparativo entre baseline e resultado atual.

    Args:
        current: EvalReport da Fase 2 com avg_faithfulness e avg_answer_relevance.
        baseline: EvalReport da Fase 2 do baseline (Fase 1 original).
        model_name: nome do modelo avaliado.

    Returns:
        Dict com métricas comparativas, deltas e flags de meta atingida.
    """

    def _metric(key: str, target: float) -> dict:
        baseline_val = baseline.get(key, 0.0)
        current_val = current.get(key, 0.0)
        return {
            "baseline": round(baseline_val, 4),
            "finetuned": round(current_val, 4),
            "delta": round(current_val - baseline_val, 4),
            "target": target,
            "achieved": current_val > target,
        }

    return {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "faithfulness": _metric("avg_faithfulness", target=0.70),
            "answer_relevance": _metric("avg_answer_relevance", target=0.70),
            # TODO: métrica ReAct format compliance (>90%) não automatizada aqui.
            # Requer rodar agente da Fase 3 com modelo fine-tuned + medir parse.
        },
        "full_result": current,
        "full_baseline": baseline,
    }


def run_evaluation(
    model_name: str = "llama3.2-finetuned",
    baseline_path: str = DEFAULT_BASELINE_PATH,
    dataset_path: str = DEFAULT_DATASET_PATH,
    output_dir: str = "data/results",
) -> dict:
    """Executa a avaliação da Fase 2 com o modelo fine-tuned e compara com baseline.

    Args:
        model_name: nome do modelo no Ollama a avaliar.
        baseline_path: diretório com resultados do baseline da Fase 2.
        dataset_path: caminho para eval_dataset.json da Fase 2.
        output_dir: diretório para salvar o relatório comparativo.

    Returns:
        Dict com comparação baseline vs. fine-tuned.
    """
    evaluate_dataset, load_dataset, save_report = _load_fase2_functions()

    logger.info("Carregando dataset de avaliação: %s", dataset_path)
    entries = load_dataset(Path(dataset_path))
    logger.info("Dataset: %d perguntas", len(entries))

    logger.info("Executando avaliação com modelo: %s", model_name)
    report = evaluate_dataset(
        entries,
        retrieve_fn=_make_retrieve_fn(),
        generate_fn=make_generate_fn(model_name),
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = f"fase4_{timestamp}"
    result_path = save_report(report, label=label)
    logger.info("Resultado salvo em: %s", result_path)

    logger.info("Carregando baseline para comparação...")
    baseline = _load_latest_baseline(baseline_path)

    comparison = _compare_results(report, baseline, model_name)

    # Salvar comparação
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    comparison_path = Path(output_dir) / f"comparison_{timestamp}.json"
    comparison_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False))
    logger.info("Comparação salva em: %s", comparison_path)

    _print_summary(comparison)

    return comparison


def _print_summary(comparison: dict) -> None:
    """Imprime sumário dos resultados no terminal."""
    metrics = comparison["metrics"]

    print("\n" + "=" * 50)
    print(f"AVALIAÇÃO: {comparison['model']}")
    print("=" * 50)
    for name, data in metrics.items():
        status = "✓" if data["achieved"] else "✗"
        sign = "+" if data["delta"] >= 0 else ""
        delta_str = f"{sign}{data['delta']:.3f}"
        print(
            f"{status} {name:20s}: {data['baseline']:.3f} → {data['finetuned']:.3f} "
            f"({delta_str}) | meta: >{data['target']:.2f}"
        )
    print("=" * 50 + "\n")


def main() -> None:
    """CLI para avaliação comparativa."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Avalia modelo fine-tuned vs. baseline"
    )
    parser.add_argument("--model", default="llama3.2-finetuned")
    parser.add_argument("--baseline-path", default=DEFAULT_BASELINE_PATH)
    parser.add_argument("--output", default="data/results")
    args = parser.parse_args()

    run_evaluation(
        model_name=args.model,
        baseline_path=args.baseline_path,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
