"""Módulo de recuperação de chunks relevantes.

Responsabilidade: dada uma query do usuário, encontrar os chunks mais
semanticamente similares no ChromaDB. Este módulo existe como camada
de separação entre a indexação (embeddings.py) e o uso dos dados (generation.py).

O retrieval converte a query em embedding e realiza busca por similaridade
cosine no ChromaDB, retornando os top-K chunks mais próximos no espaço vetorial.
A qualidade do retrieval determina diretamente a qualidade da resposta final.
"""

import chromadb

from src.embeddings import (
    EMBEDDING_MODEL,
    generate_embedding,
    get_collection,
)

# Configurações
TOP_K = 5  # número de chunks retornados por query


def retrieve(
    query: str,
    collection: chromadb.Collection | None = None,
    top_k: int = TOP_K,
    embedding_model: str = EMBEDDING_MODEL,
) -> list[dict[str, str | int | float]]:
    """Busca os chunks mais relevantes para uma query.

    Converte a query em embedding e usa a busca por similaridade do ChromaDB
    para encontrar os top_k chunks mais próximos no espaço vetorial.
    O score retornado é 1 - distance (quanto maior, mais similar).

    Args:
        query: pergunta ou texto de busca do usuário.
        collection: collection do ChromaDB (usa a padrão se None).
        top_k: número máximo de chunks a retornar.
        embedding_model: modelo Ollama usado para embedar a query.

    Returns:
        Lista de dicts com 'text', 'source', 'page' e 'score',
        ordenados do mais para o menos relevante.
    """
    if collection is None:
        collection = get_collection()

    if collection.count() == 0:
        return []

    n_results = min(top_k, collection.count())
    embedding = generate_embedding(query, model=embedding_model)

    results = collection.query(
        query_embeddings=[embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    return [
        {
            "text": doc,
            "source": meta.get("source", ""),
            "page": meta.get("page", 0),
            "score": round(1 - dist, 4),
        }
        for doc, meta, dist in zip(docs, metas, distances, strict=True)
    ]


def format_context(chunks: list[dict[str, str | int | float]]) -> str:
    """Formata os chunks recuperados em um bloco de contexto para o LLM.

    Produz uma string numerada com o texto e a fonte de cada chunk,
    separados por delimitador. Usada em generation.py para montar o prompt.

    Args:
        chunks: lista de chunks retornados por retrieve().

    Returns:
        String formatada com os trechos e suas fontes. Vazia se sem chunks.
    """
    if not chunks:
        return ""

    parts = [
        f"[{i}] Fonte: {chunk['source']} (página {chunk['page']})\n{chunk['text']}"
        for i, chunk in enumerate(chunks, 1)
    ]

    return "\n\n---\n\n".join(parts)
