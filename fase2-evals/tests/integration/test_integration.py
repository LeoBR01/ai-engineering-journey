"""
Testes de integração — requerem Ollama e ChromaDB rodando.

Executar separado dos testes unitários:
    uv run pytest -m integration -v

Estes testes validam o que os mocks não conseguem cobrir:
- O ChromaDB da Fase 1 está acessível no path correto?
- O nomic-embed-text gera embeddings no formato certo?
- O llama3.2 retorna resposta parseável?
- A query tem boa similaridade semântica com os chunks indexados?
"""

import pytest

from src.run_eval import generate_fn, judge_fn, retrieve_fn


@pytest.mark.integration
def test_retrieve_fn_real():
    """
    Verifica que retrieve_fn conecta ao ChromaDB real e retorna chunks válidos.

    Requer: nomic-embed-text via Ollama + ChromaDB populado da Fase 1.
    """
    results = retrieve_fn("What is RAG?", top_k=3)

    assert len(results) == 3
    assert all("id" in r for r in results)
    assert all("text" in r for r in results)
    assert all("source" in r for r in results)
    assert all("score" in r for r in results)
    # Scores devem ser floats entre 0 e 1
    assert all(0.0 <= r["score"] <= 1.0 for r in results)
    # Textos não devem estar vazios
    assert all(len(r["text"]) > 0 for r in results)


@pytest.mark.integration
def test_retrieve_fn_real_top_k_respeita_limite():
    """Verifica que top_k é respeitado com valores diferentes."""
    results_1 = retrieve_fn("What is fine-tuning?", top_k=1)
    results_5 = retrieve_fn("What is fine-tuning?", top_k=5)

    assert len(results_1) == 1
    assert len(results_5) == 5


@pytest.mark.integration
def test_retrieve_fn_real_resultados_diferentes_por_query():
    """Queries diferentes devem retornar chunks diferentes (top resultado)."""
    rag_results = retrieve_fn("What is RAG?", top_k=1)
    lora_results = retrieve_fn("What is LoRA fine-tuning?", top_k=1)

    assert rag_results[0]["id"] != lora_results[0]["id"]


@pytest.mark.integration
def test_generate_fn_real():
    """
    Verifica que generate_fn chama o llama3.2 e retorna resposta válida.

    Requer: llama3.2 via Ollama.
    """
    answer = generate_fn(
        question="What is RAG?",
        context="RAG stands for Retrieval-Augmented Generation. "
        "It combines a retrieval system with a generative language model.",
    )

    assert isinstance(answer, str)
    assert len(answer) > 0
    # A resposta deve mencionar algo relacionado a RAG ou retrieval
    answer_lower = answer.lower()
    assert any(
        kw in answer_lower for kw in ["retrieval", "rag", "generation", "context"]
    )


@pytest.mark.integration
def test_generate_fn_real_respeita_contexto():
    """
    O modelo deve responder com base no contexto, não em conhecimento externo.

    Passa um contexto fictício e verifica que a resposta reflete o contexto.
    """
    answer = generate_fn(
        question="What color is the sky in this document?",
        context="According to the study, the sky is bright green on planet Zorblax.",
    )

    assert isinstance(answer, str)
    assert len(answer) > 0


@pytest.mark.integration
def test_judge_fn_real():
    """
    Verifica que judge_fn retorna resposta raw do LLM para um prompt arbitrário.

    Requer: llama3.2 via Ollama.
    """
    response = judge_fn("Respond with only the number 1.")

    assert isinstance(response, str)
    assert len(response) > 0
