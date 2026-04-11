"""Testes unitários para as métricas de retrieval."""

import pytest

from src.metrics_retrieval import (
    mean_recall_at_k,
    mean_reciprocal_rank,
    recall_at_k,
    reciprocal_rank,
)

# ── recall_at_k ──


def test_recall_at_k_perfeito():
    retrieved = ["a", "b", "c", "d", "e"]
    relevant = ["a", "b"]
    assert recall_at_k(retrieved, relevant, k=3) == 1.0


def test_recall_at_k_parcial():
    retrieved = ["a", "x", "y", "b", "z"]
    relevant = ["a", "b"]
    # Apenas "a" está nos top-3
    assert recall_at_k(retrieved, relevant, k=3) == 0.5


def test_recall_at_k_zero():
    retrieved = ["x", "y", "z"]
    relevant = ["a", "b"]
    assert recall_at_k(retrieved, relevant, k=3) == 0.0


def test_recall_at_k_sem_relevantes():
    assert recall_at_k(["a", "b"], [], k=3) == 0.0


def test_recall_at_k_respeita_limite():
    """Chunks relevantes além de k não devem ser contados."""
    retrieved = ["x", "y", "a"]  # "a" está na posição 3
    relevant = ["a"]
    assert recall_at_k(retrieved, relevant, k=2) == 0.0
    assert recall_at_k(retrieved, relevant, k=3) == 1.0


# ── reciprocal_rank ──


def test_reciprocal_rank_primeiro():
    assert reciprocal_rank(["a", "b", "c"], ["a"]) == 1.0


def test_reciprocal_rank_segundo():
    assert reciprocal_rank(["x", "a", "c"], ["a"]) == pytest.approx(0.5)


def test_reciprocal_rank_terceiro():
    assert reciprocal_rank(["x", "y", "a"], ["a"]) == pytest.approx(1 / 3)


def test_reciprocal_rank_nao_encontrado():
    assert reciprocal_rank(["x", "y", "z"], ["a"]) == 0.0


def test_reciprocal_rank_multiplos_relevantes():
    """Deve parar no primeiro relevante encontrado."""
    assert reciprocal_rank(["x", "a", "b"], ["a", "b"]) == pytest.approx(0.5)


# ── médias ──


def test_mean_recall_at_k_vazio():
    assert mean_recall_at_k([], k=5) == 0.0


def test_mean_recall_at_k_calculo():
    results = [
        (["a", "b", "c"], ["a"]),  # recall@3 = 1.0
        (["x", "y", "z"], ["a"]),  # recall@3 = 0.0
    ]
    assert mean_recall_at_k(results, k=3) == pytest.approx(0.5)


def test_mean_reciprocal_rank_vazio():
    assert mean_reciprocal_rank([]) == 0.0


def test_mean_reciprocal_rank_calculo():
    results = [
        (["a", "b"], ["a"]),  # RR = 1.0
        (["x", "a"], ["a"]),  # RR = 0.5
    ]
    assert mean_reciprocal_rank(results) == pytest.approx(0.75)
