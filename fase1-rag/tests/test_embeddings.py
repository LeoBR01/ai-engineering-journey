"""Testes do módulo de embeddings (src/embeddings.py).

Verifica a geração de embeddings, a criação de collections no ChromaDB
e a persistência dos chunks. Usa mocks para Ollama e ChromaDB para evitar
dependências externas em ambiente de CI.
"""

from unittest.mock import MagicMock, patch

from src.embeddings import (
    generate_embedding,
    get_chroma_collection,
    store_chunks,
)

SAMPLE_CHUNKS = [
    {"text": "texto um", "source": "doc1.pdf", "page": 1, "chunk_id": "id-1"},
    {"text": "texto dois", "source": "doc2.pdf", "page": 2, "chunk_id": "id-2"},
]

FAKE_VEC = [0.1, 0.2, 0.3]


def _mock_embed_response(vec: list[float]) -> MagicMock:
    """Cria um EmbedResponse falso compatível com ollama.embed()."""
    resp = MagicMock()
    resp.embeddings = [vec]
    return resp


class TestGenerateEmbedding:
    def test_retorna_lista_de_floats(self):
        with patch(
            "src.embeddings.ollama.embed",
            return_value=_mock_embed_response(FAKE_VEC),
        ):
            result = generate_embedding("texto qualquer")
        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)

    def test_chama_ollama_com_modelo_correto(self):
        with patch(
            "src.embeddings.ollama.embed",
            return_value=_mock_embed_response(FAKE_VEC),
        ) as mock_embed:
            generate_embedding("texto", model="nomic-embed-text")
        mock_embed.assert_called_once_with(model="nomic-embed-text", input="texto")

    def test_texto_vazio_nao_levanta_excecao(self):
        with patch(
            "src.embeddings.ollama.embed",
            return_value=_mock_embed_response([0.0, 0.0]),
        ):
            result = generate_embedding("")
        assert isinstance(result, list)


class TestGetChromaCollection:
    def test_cria_collection_se_nao_existir(self):
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection

        with patch(
            "src.embeddings.chromadb.PersistentClient", return_value=mock_client
        ):
            col = get_chroma_collection("nova_collection")

        mock_client.get_or_create_collection.assert_called_once_with(
            name="nova_collection"
        )
        assert col is mock_collection

    def test_retorna_collection_existente(self):
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection

        with patch(
            "src.embeddings.chromadb.PersistentClient", return_value=mock_client
        ):
            col = get_chroma_collection()

        assert col is mock_collection


class TestStoreChunks:
    def test_adiciona_todos_os_chunks_na_collection(self):
        mock_collection = MagicMock()

        with patch("src.embeddings.generate_embedding", return_value=FAKE_VEC):
            store_chunks(SAMPLE_CHUNKS, mock_collection)

        mock_collection.add.assert_called_once()
        kwargs = mock_collection.add.call_args.kwargs
        assert len(kwargs["ids"]) == 2
        assert kwargs["ids"] == ["id-1", "id-2"]

    def test_lista_vazia_nao_chama_add(self):
        mock_collection = MagicMock()
        store_chunks([], mock_collection)
        mock_collection.add.assert_not_called()

    def test_metadados_de_source_sao_preservados(self):
        mock_collection = MagicMock()

        with patch("src.embeddings.generate_embedding", return_value=FAKE_VEC):
            store_chunks(SAMPLE_CHUNKS, mock_collection)

        metadatas = mock_collection.add.call_args.kwargs["metadatas"]
        assert metadatas[0]["source"] == "doc1.pdf"
        assert metadatas[0]["page"] == 1
        assert metadatas[1]["source"] == "doc2.pdf"
        assert metadatas[1]["page"] == 2
