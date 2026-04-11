"""
Geração e persistência de relatórios de avaliação.

Formata os resultados do EvalReport em texto legível e salva em
data/results/ com timestamp para comparação entre experimentos.
"""

import json
from datetime import datetime
from pathlib import Path

from .evaluator import EvalReport

RESULTS_DIR = Path(__file__).parent.parent / "data" / "results"


def format_report(report: EvalReport) -> str:
    """
    Formata o relatório de avaliação como texto legível.

    Args:
        report: Resultado da avaliação completa.

    Returns:
        String formatada com métricas agregadas e detalhes por entrada.
    """
    lines = [
        "=" * 60,
        "RELATÓRIO DE AVALIAÇÃO DO PIPELINE RAG",
        "=" * 60,
        f"Total de perguntas avaliadas: {report['total_entries']}",
        "",
        "── MÉTRICAS DE RETRIEVAL ──",
        f"  Recall@3 médio : {report['avg_recall_at_3']:.3f}",
        f"  Recall@5 médio : {report['avg_recall_at_5']:.3f}",
        f"  MRR médio      : {report['avg_mrr']:.3f}",
        "",
        "── MÉTRICAS DE GERAÇÃO (LLM-as-judge) ──",
        f"  Faithfulness médio     : {report['avg_faithfulness']:.3f}",
        f"  Answer Relevance médio : {report['avg_answer_relevance']:.3f}",
        "",
        "── DETALHES POR PERGUNTA ──",
    ]

    for entry in report["entries"]:
        lines += [
            f"\n[{entry['id']}] {entry['question'][:80]}",
            f"  Resposta esperada : {entry['expected_answer'][:100]}",
            f"  Resposta gerada   : {entry['generated_answer'][:100]}",
            f"  Recall@3={entry['recall_at_3']:.2f} | "
            f"Recall@5={entry['recall_at_5']:.2f} | "
            f"RR={entry['mrr']:.2f} | "
            f"Faith={entry['faithfulness']:.2f} | "
            f"Relev={entry['answer_relevance']:.2f}",
        ]

    lines.append("=" * 60)
    return "\n".join(lines)


def save_report(report: EvalReport, label: str = "") -> Path:
    """
    Persiste o relatório em data/results/ como JSON e texto.

    O nome do arquivo inclui timestamp para facilitar comparação
    entre diferentes experimentos e configurações.

    Args:
        report: Resultado da avaliação completa.
        label: Identificador opcional para o experimento (ex: "chunking_512").

    Returns:
        Caminho do arquivo JSON salvo.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = f"_{label}" if label else ""
    base_name = f"eval{slug}_{timestamp}"

    json_path = RESULTS_DIR / f"{base_name}.json"
    txt_path = RESULTS_DIR / f"{base_name}.txt"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(format_report(report))

    return json_path
