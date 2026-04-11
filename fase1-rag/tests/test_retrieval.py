"""Testes do módulo de retrieval (src/retrieval.py).

Verifica a busca semântica no ChromaDB e a formatação do contexto.
Usa mocks para ChromaDB e para a geração de embeddings, garantindo
que os testes não dependam de serviços externos.
"""

from unittest.mock import MagicMock, patch

from src.retrieval import format_context, retrieve

FAKE_EMBEDDING = [0.1, 0.2, 0.3]

CHROMA_RESULT = {
    "ids": [["id-1", "id-2"]],
    "documents": [["texto do chunk um", "texto do chunk dois"]],
    "metadatas": [[{"source": "doc.pdf", "page": 1}, {"source": "doc.pdf", "page": 2}]],
    "distances": [[0.1, 0.3]],
}

SAMPLE_CHUNKS = [
    {"text": "trecho um", "source": "doc1.pdf", "page": 1, "score": 0.9},
    {"text": "trecho dois", "source": "doc2.pdf", "page": 3, "score": 0.8},
]


def _mock_collection(count: int = 5, result: dict = CHROMA_RESULT) -> MagicMock:
    """Cria uma collection ChromaDB falsa com count e query configurados."""
    col = MagicMock()
    col.count.return_value = count
    col.query.return_value = result
    return col


class TestRetrieve:
    def test_retorna_top_k_chunks(self):
        with patch("src.retrieval.generate_embedding", return_value=FAKE_EMBEDDING):
            result = retrieve("minha query", collection=_mock_collection(), top_k=2)
        assert len(result) == 2

    def test_chunks_incluem_campo_distance(self):
        with patch("src.retrieval.generate_embedding", return_value=FAKE_EMBEDDING):
            result = retrieve("query", collection=_mock_collection())
        for chunk in result:
            assert "score" in chunk
            assert "text" in chunk
            assert "source" in chunk
            assert "page" in chunk

    def test_query_vazia_nao_levanta_excecao(self):
        with patch("src.retrieval.generate_embedding", return_value=FAKE_EMBEDDING):
            result = retrieve("", collection=_mock_collection())
        assert isinstance(result, list)

    def test_collection_vazia_retorna_lista_vazia(self):
        result = retrieve("qualquer coisa", collection=_mock_collection(count=0))
        assert result == []


class TestFormatContext:
    def test_inclui_fonte_de_cada_chunk(self):
        context = format_context(SAMPLE_CHUNKS)
        assert "doc1.pdf" in context
        assert "doc2.pdf" in context

    def test_lista_vazia_retorna_string_vazia(self):
        assert format_context([]) == ""

    def test_chunks_separados_por_delimitador(self):
        context = format_context(SAMPLE_CHUNKS)
        assert "---" in context
