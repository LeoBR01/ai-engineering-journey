# src/cache.py
"""Cache semântico usando ChromaDB com similaridade cosine.

Armazena respostas indexadas pelo embedding da query.
Retorna hit se similaridade cosine > CACHE_SIMILARITY_THRESHOLD e não expirada.
TTL de 24h verificado via metadata cached_at.
"""

import logging
import time
from pathlib import Path

import chromadb
import ollama

logger = logging.getLogger(__name__)

CACHE_SIMILARITY_THRESHOLD: float = 0.92
CACHE_TTL_SECONDS: int = 24 * 3600
EMBED_MODEL: str = "nomic-embed-text"
CACHE_COLLECTION: str = "query_cache"
CHROMA_PATH: Path = Path(__file__).resolve().parent.parent / "data" / "chroma_cache"


class SemanticCache:
    """Cache semântico: armazena respostas indexadas por embedding da query."""

    def __init__(
        self,
        chroma_path: Path = CHROMA_PATH,
        threshold: float = CACHE_SIMILARITY_THRESHOLD,
        ttl: int = CACHE_TTL_SECONDS,
    ) -> None:
        self._threshold = threshold
        self._ttl = ttl
        client = chromadb.PersistentClient(path=str(chroma_path))
        self._collection = client.get_or_create_collection(
            name=CACHE_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    def _embed(self, text: str) -> list[float]:
        resp = ollama.embeddings(model=EMBED_MODEL, prompt=text)
        return resp["embedding"]

    def get(self, question: str) -> str | None:
        """Retorna resposta cached se similaridade > threshold e não expirada.

        Args:
            question: pergunta do usuário.

        Returns:
            Resposta cached como string, ou None se miss/expirado.
        """
        try:
            embedding = self._embed(question)
            results = self._collection.query(
                query_embeddings=[embedding],
                n_results=1,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            return None

        if not results["distances"] or not results["distances"][0]:
            return None

        distance = results["distances"][0][0]
        similarity = 1.0 - distance

        if similarity < self._threshold:
            return None

        metadata = results["metadatas"][0][0]
        cached_at = float(metadata.get("cached_at", 0.0))
        if time.time() - cached_at > self._ttl:
            return None

        logger.info("cache_hit similarity=%.3f", similarity)
        return results["documents"][0][0]

    def set(self, question: str, answer: str) -> None:
        """Persiste uma resposta no cache.

        Args:
            question: pergunta original.
            answer: resposta a armazenar.
        """
        try:
            embedding = self._embed(question)
            doc_id = f"q_{abs(hash(question)) % (10**9)}"
            self._collection.upsert(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[answer],
                metadatas=[{"question": question, "cached_at": time.time()}],
            )
        except Exception as exc:
            logger.warning("cache_set_failed error=%s", exc)
