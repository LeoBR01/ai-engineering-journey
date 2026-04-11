"""Testes do orquestrador do pipeline (src/pipeline.py).

Verifica que os fluxos de indexação e query chamam os módulos corretos
na ordem esperada. Todos os módulos dependentes são mockados — o objetivo
aqui é testar a orquestração, não a lógica de cada etapa individual.
"""

from unittest.mock import MagicMock, patch

from src.pipeline import index, query

SAMPLE_CHUNKS = [
    {"text": "trecho A", "source": "paper1.pdf", "page": 1, "chunk_id": "id-1"},
    {"text": "trecho B", "source": "paper2.pdf", "page": 3, "chunk_id": "id-2"},
]

RETRIEVED_CHUNKS = [
    {"text": "trecho A", "source": "paper1.pdf", "page": 1, "score": 0.92},
]


class TestIndex:
    def test_chama_ingest_pdfs(self, tmp_path):
        with (
            patch("src.pipeline.ingest_pdfs", return_value=[]) as mock_ingest,
            patch("src.pipeline.get_chroma_collection", return_value=MagicMock()),
            patch("src.pipeline.store_chunks"),
        ):
            index(tmp_path)
        mock_ingest.assert_called_once_with(tmp_path)

    def test_chama_store_chunks_com_resultado_da_ingestao(self, tmp_path):
        mock_col = MagicMock()
        with (
            patch("src.pipeline.ingest_pdfs", return_value=SAMPLE_CHUNKS),
            patch("src.pipeline.get_chroma_collection", return_value=mock_col),
            patch("src.pipeline.store_chunks") as mock_store,
        ):
            index(tmp_path)
        mock_store.assert_called_once_with(SAMPLE_CHUNKS, mock_col)

    def test_diretorio_sem_pdfs_nao_levanta_excecao(self, tmp_path):
        with (
            patch("src.pipeline.ingest_pdfs", return_value=[]),
            patch("src.pipeline.get_chroma_collection", return_value=MagicMock()),
            patch("src.pipeline.store_chunks") as mock_store,
        ):
            index(tmp_path)  # não deve levantar exceção
        mock_store.assert_not_called()


class TestQuery:
    def test_retorna_string(self):
        with (
            patch("src.pipeline.get_chroma_collection", return_value=MagicMock()),
            patch("src.pipeline.retrieve", return_value=RETRIEVED_CHUNKS),
            patch("src.pipeline.generate", return_value="Resposta do LLM"),
        ):
            result = query("O que é atenção?")
        assert isinstance(result, str)
        assert result == "Resposta do LLM"

    def test_chama_retrieve_com_a_pergunta(self):
        mock_col = MagicMock()
        with (
            patch("src.pipeline.get_chroma_collection", return_value=mock_col),
            patch("src.pipeline.retrieve", return_value=[]) as mock_retrieve,
            patch("src.pipeline.generate", return_value="ok"),
        ):
            query("minha pergunta")
        mock_retrieve.assert_called_once_with("minha pergunta", collection=mock_col)

    def test_chama_generate_com_contexto_e_pergunta(self):
        with (
            patch("src.pipeline.get_chroma_collection", return_value=MagicMock()),
            patch("src.pipeline.retrieve", return_value=RETRIEVED_CHUNKS),
            patch("src.pipeline.generate", return_value="ok") as mock_gen,
        ):
            query("pergunta de teste")
        mock_gen.assert_called_once_with("pergunta de teste", RETRIEVED_CHUNKS)

    def test_fluxo_completo_em_ordem(self):
        call_order: list[str] = []

        def mock_retrieve(*args, **kwargs):
            call_order.append("retrieve")
            return RETRIEVED_CHUNKS

        def mock_generate(*args, **kwargs):
            call_order.append("generate")
            return "resposta"

        with (
            patch("src.pipeline.get_chroma_collection", return_value=MagicMock()),
            patch("src.pipeline.retrieve", mock_retrieve),
            patch("src.pipeline.generate", mock_generate),
        ):
            query("qualquer pergunta")

        assert call_order == ["retrieve", "generate"]
