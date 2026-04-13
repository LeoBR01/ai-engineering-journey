"""Testes de integração para a Fase 4.

Requerem: Ollama rodando com llama3.2, ChromaDB da Fase 1 populado.
Executar com: pytest tests/test_integration.py -m integration -v
"""

import json
from pathlib import Path

import pytest

FASE1_CHROMA = Path(__file__).resolve().parent.parent.parent / "fase1-rag" / "chroma_db"


@pytest.mark.integration
def test_rag_pair_generation_with_real_chromadb() -> None:
    """Gera 3 pares RAG reais a partir do ChromaDB da Fase 1."""
    import tempfile

    from src.generate_dataset import generate_rag_dataset

    assert FASE1_CHROMA.exists(), f"ChromaDB não encontrado: {FASE1_CHROMA}"

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "rag_pairs.jsonl"
        n_pairs = generate_rag_dataset(
            str(output_path),
            chroma_path=str(FASE1_CHROMA),
            max_pairs=3,
            sample_every=100,
        )

        assert n_pairs >= 1
        assert output_path.exists()


@pytest.mark.integration
def test_react_cycle_generation_with_real_tools() -> None:
    """Gera 2 ciclos ReAct reais usando ChromaDB e Ollama."""
    import tempfile

    from src.generate_dataset import generate_react_dataset

    assert FASE1_CHROMA.exists(), f"ChromaDB não encontrado: {FASE1_CHROMA}"

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "react_cycles.jsonl"
        questions = [
            "What are the main advantages of retrieval-augmented generation?",
            "How does the attention mechanism work?",
        ]
        n_cycles = generate_react_dataset(str(output_path), questions=questions)

        assert n_cycles == 2
        assert output_path.exists()

        # Verificar formato correto dos ciclos
        cycles = [
            json.loads(line) for line in output_path.read_text().splitlines() if line
        ]
        for cycle in cycles:
            assistant_content = cycle["messages"][2]["content"]
            assert "Thought:" in assistant_content
            assert "Action: search_papers" in assistant_content
            assert "Final Answer:" in assistant_content
