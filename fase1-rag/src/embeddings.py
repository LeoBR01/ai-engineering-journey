"""Módulo de geração e persistência de embeddings.

Responsabilidade: converter chunks de texto em vetores numéricos (embeddings)
e armazená-los no ChromaDB. Este módulo existe porque a busca semântica
opera no espaço vetorial — o ChromaDB encontra chunks similares calculando
a distância entre vetores, não por correspondência exata de palavras.

Os embeddings são gerados pelo modelo `nomic-embed-text` via Ollama,
garantindo que tudo rode localmente sem chamadas a APIs externas.
"""

import chromadb
import ollama

# Configurações
EMBEDDING_MODEL = "nomic-embed-text"
CHROMA_COLLECTION = "papers"
CHROMA_PATH = "./chroma_db"


def get_chroma_collection(
    collection_name: str = CHROMA_COLLECTION,
) -> chromadb.Collection:
    """Cria ou recupera uma collection persistente no ChromaDB.

    Usa PersistentClient para que os dados sobrevivam entre execuções.
    Se a collection não existir, ela é criada automaticamente.

    Args:
        collection_name: nome da collection.

    Returns:
        Collection do ChromaDB pronta para uso.
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(name=collection_name)


def generate_embedding(text: str, model: str = EMBEDDING_MODEL) -> list[float]:
    """Gera o embedding de um texto usando Ollama.

    Args:
        text: texto a ser convertido em vetor.
        model: nome do modelo de embedding no Ollama.

    Returns:
        Vetor de floats representando o texto no espaço semântico.
    """
    response = ollama.embed(model=model, input=text)
    return response.embeddings[0]


def store_chunks(
    chunks: list[dict[str, str | int]],
    collection: chromadb.Collection,
) -> None:
    """Gera embeddings e armazena chunks no ChromaDB em lote.

    Cada chunk vira um documento no ChromaDB com seu vetor de embedding,
    o texto original e metadados (arquivo de origem e página).

    Args:
        chunks: lista de dicts com 'text', 'source', 'page' e 'chunk_id'.
        collection: collection do ChromaDB onde os chunks serão salvos.
    """
    if not chunks:
        return

    embeddings = [generate_embedding(c["text"]) for c in chunks]

    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=embeddings,
        metadatas=[{"source": c["source"], "page": c.get("page", 0)} for c in chunks],
    )


def get_collection() -> chromadb.Collection:
    """Retorna a collection padrão do ChromaDB.

    Atalho para get_chroma_collection() com o nome padrão configurado.

    Returns:
        Collection 'papers' do ChromaDB.
    """
    return get_chroma_collection()


def index_documents(chunks: list[dict[str, str | int]]) -> None:
    """Indexa uma lista de chunks no ChromaDB.

    Ponto de entrada principal do módulo: obtém a collection e
    persiste todos os chunks com seus embeddings.

    Args:
        chunks: lista de dicts gerada por ingestion.ingest_pdfs().
    """
    collection = get_chroma_collection()
    store_chunks(chunks, collection)
