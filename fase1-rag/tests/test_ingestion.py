"""Testes do módulo de ingestão (src/ingestion.py).

Verifica leitura de PDFs, chunking com e sem overlap, e o comportamento
do ingest_pdfs ao lidar com diretórios vazios ou com múltiplos arquivos.
Não depende de Ollama nem ChromaDB — usa apenas arquivos locais.
"""

from pathlib import Path

import fitz
import pytest

from src.ingestion import chunk_text, ingest_pdfs, load_pdf


def _create_pdf(path: Path, text: str) -> None:
    """Cria um PDF de uma página com o texto fornecido."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), text)
    doc.save(str(path))
    doc.close()


class TestChunkText:
    def test_chunks_respeitam_tamanho_maximo(self):
        text = "a" * 2000
        chunks = chunk_text(text, chunk_size=512, overlap=50)
        assert all(len(c) <= 512 for c in chunks)

    def test_overlap_entre_chunks_consecutivos(self):
        text = "a" * 600
        chunks = chunk_text(text, chunk_size=512, overlap=50)
        assert len(chunks) >= 2
        # O final do primeiro chunk deve coincidir com o início do segundo
        assert chunks[0][-50:] == chunks[1][:50]

    def test_texto_menor_que_chunk_retorna_unico_chunk(self):
        text = "texto curto"
        chunks = chunk_text(text, chunk_size=512, overlap=50)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_texto_vazio_retorna_lista_vazia(self):
        assert chunk_text("") == []


class TestLoadPdf:
    def test_extrai_texto_de_pdf_valido(self, tmp_path):
        pdf_path = tmp_path / "test.pdf"
        _create_pdf(pdf_path, "Conteúdo de teste do PDF.")
        text = load_pdf(pdf_path)
        assert "Conteúdo de teste do PDF." in text

    def test_levanta_erro_para_arquivo_inexistente(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_pdf(tmp_path / "inexistente.pdf")


class TestIngestPdfs:
    def test_retorna_lista_vazia_para_diretorio_sem_pdfs(self, tmp_path):
        result = ingest_pdfs(tmp_path)
        assert result == []

    def test_cada_chunk_tem_campos_obrigatorios(self, tmp_path):
        _create_pdf(tmp_path / "doc.pdf", "Texto longo " * 50)
        chunks = ingest_pdfs(tmp_path)
        assert len(chunks) > 0
        for chunk in chunks:
            assert "text" in chunk
            assert "source" in chunk
            assert "page" in chunk
            assert "chunk_id" in chunk
            assert chunk["source"] == "doc.pdf"
            assert chunk["page"] == 1

    def test_chunk_id_e_unico_por_documento(self, tmp_path):
        _create_pdf(tmp_path / "doc.pdf", "Texto longo " * 200)
        chunks = ingest_pdfs(tmp_path)
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids))
