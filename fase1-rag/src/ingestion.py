"""Módulo de ingestão de documentos PDF.

Responsabilidade: ler arquivos PDF da pasta data/pdfs/ e dividir o texto
em chunks menores (chunking). Este módulo existe porque LLMs têm janela de
contexto limitada — não é possível enviar um PDF inteiro como contexto.
O chunking garante que apenas os trechos mais relevantes sejam recuperados
e usados na geração.

Estratégia de chunking: janela deslizante com overlap para evitar que
informações importantes fiquem cortadas nas bordas dos chunks.
"""

import uuid
from pathlib import Path

import fitz  # PyMuPDF

# Configurações de chunking
CHUNK_SIZE = 512  # número de caracteres por chunk
CHUNK_OVERLAP = 50  # sobreposição entre chunks consecutivos
PDF_DIR = Path("data/pdfs")


def load_pdf(path: Path) -> str:
    """Extrai todo o texto de um arquivo PDF, concatenando todas as páginas.

    Args:
        path: caminho para o arquivo PDF.

    Returns:
        Texto completo extraído do PDF (todas as páginas concatenadas).

    Raises:
        FileNotFoundError: se o arquivo não existir.
    """
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    doc = fitz.open(str(path))
    return "\n".join(page.get_text() for page in doc)


def chunk_text(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Divide um texto em chunks menores com overlap (janela deslizante).

    O overlap garante que informações nas bordas entre dois chunks
    consecutivos não se percam — os últimos `overlap` caracteres de um
    chunk também aparecem no início do próximo.

    Args:
        text: texto a ser dividido.
        chunk_size: tamanho máximo de cada chunk em caracteres.
        overlap: número de caracteres de sobreposição entre chunks.

    Returns:
        Lista de chunks de texto. Retorna lista vazia se o texto for vazio.
    """
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    step = chunk_size - overlap

    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += step

    return chunks


def ingest_pdfs(pdf_dir: Path = PDF_DIR) -> list[dict[str, str | int]]:
    """Lê todos os PDFs de um diretório e retorna chunks com metadados.

    Processa cada PDF página por página, aplica chunking em cada página
    e adiciona metadados (arquivo de origem, número da página, id único).

    Args:
        pdf_dir: diretório contendo os arquivos PDF.

    Returns:
        Lista de dicts com as chaves:
        - 'text': conteúdo do chunk
        - 'source': nome do arquivo PDF
        - 'page': número da página (começa em 1)
        - 'chunk_id': identificador único UUID (string)
    """
    results: list[dict[str, str | int]] = []

    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        doc = fitz.open(str(pdf_path))
        for page_num, page in enumerate(doc, start=1):
            page_text = page.get_text()
            for chunk in chunk_text(page_text):
                results.append(
                    {
                        "text": chunk,
                        "source": pdf_path.name,
                        "page": page_num,
                        "chunk_id": str(uuid.uuid4()),
                    }
                )

    return results
