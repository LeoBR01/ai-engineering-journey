"""Orquestrador do pipeline RAG completo.

Responsabilidade: coordenar os quatro módulos (ingestion → embeddings →
retrieval → generation) em dois fluxos distintos:

1. **Indexação** (`index`): processa PDFs e popula o ChromaDB. Deve ser
   executado uma vez (ou quando novos documentos são adicionados).

2. **Query** (`query`): recebe uma pergunta, recupera contexto relevante
   e retorna a resposta gerada pelo LLM.

Este módulo existe para separar a interface de uso da lógica de cada etapa.
Não contém lógica de negócio — apenas chama as funções dos outros módulos
na ordem correta e passa os dados entre elas.
"""

import sys
from pathlib import Path

# Garante que a raiz do projeto está em sys.path quando o script é executado
# diretamente (python src/pipeline.py). Sem efeito ao usar `python -m src.pipeline`.
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.embeddings import get_chroma_collection, store_chunks  # noqa: E402
from src.generation import generate  # noqa: E402
from src.ingestion import PDF_DIR, ingest_pdfs  # noqa: E402
from src.retrieval import retrieve  # noqa: E402


def index(pdf_dir: Path = PDF_DIR) -> None:
    """Indexa todos os PDFs do diretório no ChromaDB.

    Executa: ingestão de PDFs → geração de embeddings → persistência no ChromaDB.
    Deve ser chamado antes de qualquer query, e re-executado quando novos
    PDFs forem adicionados.

    Args:
        pdf_dir: diretório contendo os arquivos PDF a indexar.
    """
    print(f"Lendo PDFs de {pdf_dir} ...")
    chunks = ingest_pdfs(pdf_dir)

    if not chunks:
        print("Nenhum PDF encontrado. Verifique o diretório:", pdf_dir)
        return

    pdf_count = len({c["source"] for c in chunks})
    print(f"  {pdf_count} PDFs processados | {len(chunks)} chunks gerados")

    print("Gerando embeddings e indexando no ChromaDB ...")
    collection = get_chroma_collection()
    store_chunks(chunks, collection)
    print(f"Concluído: {len(chunks)} chunks em '{collection.name}'.")


def query(question: str) -> str:
    """Responde uma pergunta usando o pipeline RAG completo.

    Executa: embedding da query → retrieval → geração da resposta.
    Exibe as fontes consultadas (arquivo e página) antes de retornar.

    Args:
        question: pergunta do usuário em linguagem natural.

    Returns:
        Resposta gerada pelo LLM com base nos documentos indexados.
    """
    collection = get_chroma_collection()
    chunks = retrieve(question, collection=collection)

    if chunks:
        sources = sorted({(c["source"], c["page"]) for c in chunks})
        print("Fontes consultadas:")
        for source, page in sources:
            print(f"  [{source} — p. {page}]")
    else:
        print("Nenhum chunk relevante encontrado para a pergunta.")

    return generate(question, chunks)


if __name__ == "__main__":
    import sys

    USAGE = (
        "Uso:\n"
        "  python src/pipeline.py index\n"
        '  python src/pipeline.py query "sua pergunta aqui"'
    )

    if len(sys.argv) < 2 or sys.argv[1] not in ("index", "query"):
        print(USAGE)
        sys.exit(1)

    if sys.argv[1] == "index":
        index()
    else:
        if len(sys.argv) < 3:
            print("Erro: forneça uma pergunta após 'query'.\n")
            print(USAGE)
            sys.exit(1)
        pergunta = " ".join(sys.argv[2:])
        resposta = query(pergunta)
        print("\nResposta:")
        print(resposta)
