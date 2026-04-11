"""
Métricas de avaliação do retrieval.

Mede se o retrieval está encontrando os chunks corretos:
- Recall@K: o chunk relevante está entre os K primeiros retornados?
- MRR (Mean Reciprocal Rank): em média, em que posição o chunk correto aparece?
"""


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """
    Calcula o Recall@K para uma única query.

    Recall@K = (relevantes encontrados nos top-K) / (total de relevantes)

    Args:
        retrieved_ids: IDs dos chunks retornados pelo retrieval, em ordem.
        relevant_ids: IDs dos chunks que deveriam ser retornados.
        k: Número de resultados a considerar.

    Returns:
        Float entre 0.0 e 1.0.
    """
    if not relevant_ids:
        return 0.0

    top_k = set(retrieved_ids[:k])
    relevantes_encontrados = len(top_k & set(relevant_ids))
    return relevantes_encontrados / len(relevant_ids)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """
    Calcula o Reciprocal Rank para uma única query.

    RR = 1 / posição_do_primeiro_relevante (0.0 se não encontrar nenhum)

    Args:
        retrieved_ids: IDs dos chunks retornados, em ordem de relevância.
        relevant_ids: IDs dos chunks que deveriam ser retornados.

    Returns:
        Float entre 0.0 e 1.0.
    """
    relevant_set = set(relevant_ids)
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_set:
            return 1.0 / rank
    return 0.0


def mean_recall_at_k(results: list[tuple[list[str], list[str]]], k: int) -> float:
    """
    Calcula o Recall@K médio sobre múltiplas queries.

    Args:
        results: Lista de (retrieved_ids, relevant_ids) por query.
        k: Número de resultados a considerar.

    Returns:
        Float entre 0.0 e 1.0.
    """
    if not results:
        return 0.0
    scores = [recall_at_k(retrieved, relevant, k) for retrieved, relevant in results]
    return sum(scores) / len(scores)


def mean_reciprocal_rank(results: list[tuple[list[str], list[str]]]) -> float:
    """
    Calcula o MRR (Mean Reciprocal Rank) sobre múltiplas queries.

    Args:
        results: Lista de (retrieved_ids, relevant_ids) por query.

    Returns:
        Float entre 0.0 e 1.0.
    """
    if not results:
        return 0.0
    scores = [reciprocal_rank(retrieved, relevant) for retrieved, relevant in results]
    return sum(scores) / len(scores)
