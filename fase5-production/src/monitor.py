"""Monitor de qualidade em produção via sampling e BackgroundTask.

Amostra EVAL_SAMPLE_RATE (10%) dos requests e executa answer_relevance
da Fase 2 em background — sem bloquear a resposta ao usuário.
"""

import json
import logging
import random
import sys
from pathlib import Path

from fastapi import BackgroundTasks

logger = logging.getLogger(__name__)

EVAL_SAMPLE_RATE: float = 0.1

_FASE2_ROOT = Path(__file__).resolve().parent.parent.parent / "fase2-evals"
if str(_FASE2_ROOT) not in sys.path:
    sys.path.insert(0, str(_FASE2_ROOT))

# Mesma técnica: resolve conflito de namespace src
_saved_src_modules = {
    k: v for k, v in sys.modules.items() if k == "src" or k.startswith("src.")
}
for _k in list(_saved_src_modules.keys()):
    del sys.modules[_k]

try:
    from src.metrics_generation import (  # noqa: E402
        answer_relevance as _answer_relevance,
    )

    _FASE2_AVAILABLE = True
except ImportError:
    _answer_relevance = None  # type: ignore[assignment]
    _FASE2_AVAILABLE = False
finally:
    sys.modules.update(_saved_src_modules)


def maybe_schedule_eval(
    question: str,
    answer: str,
    background_tasks: BackgroundTasks,
    trace_id: str,
) -> None:
    """Enfileira eval em background com probabilidade EVAL_SAMPLE_RATE.

    Args:
        question: pergunta original do usuário.
        answer: resposta gerada pelo pipeline.
        background_tasks: instância de BackgroundTasks do FastAPI.
        trace_id: identificador do request para correlação nos logs.
    """
    if random.random() < EVAL_SAMPLE_RATE:
        background_tasks.add_task(_run_eval, question, answer, trace_id)


def _run_eval(question: str, answer: str, trace_id: str) -> None:
    """Executa answer_relevance da Fase 2 e loga o resultado.

    Args:
        question: pergunta original.
        answer: resposta gerada.
        trace_id: identificador do request para correlação.
    """
    if not _FASE2_AVAILABLE or _answer_relevance is None:
        logger.warning("eval_skipped trace_id=%s reason=fase2_not_available", trace_id)
        return

    try:
        relevance = _answer_relevance(answer=answer, question=question)
        logger.info(
            json.dumps(
                {
                    "trace_id": trace_id,
                    "eval": True,
                    "answer_relevance": round(relevance, 3),
                }
            )
        )
    except Exception as exc:
        logger.warning("eval_failed trace_id=%s error=%s", trace_id, exc)
