# fase1-rag — Pipeline RAG Local

Pipeline **RAG (Retrieval-Augmented Generation)** 100% local para responder perguntas sobre PDFs, sem dependência de APIs pagas. LLM via Ollama, vector store via ChromaDB, leitura de PDFs via PyMuPDF.

```
Pergunta do usuário
       │
       ▼
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  Embedding  │────▶│   ChromaDB   │────▶│  Top-K chunks│
│  da pergunta│     │ (vector store│     │  relevantes  │
└─────────────┘     └──────────────┘     └──────┬───────┘
                                                 │
                                    ┌────────────▼────────────┐
                                    │  Prompt = sistema +     │
                                    │  contexto + pergunta    │
                                    │  → Ollama (llama3.2)    │
                                    └────────────┬────────────┘
                                                 │
                                          Resposta final
```

## Stack

| Componente | Tecnologia |
|---|---|
| LLM local | [Ollama](https://ollama.com/) + llama3.2 |
| Embeddings | [Ollama](https://ollama.com/) + nomic-embed-text |
| Vector store | [ChromaDB](https://www.trychroma.com/) |
| Leitura de PDF | [PyMuPDF](https://pymupdf.readthedocs.io/) |
| Gerenciamento de ambiente | [uv](https://docs.astral.sh/uv/) |
| Lint/formatação | [ruff](https://docs.astral.sh/ruff/) |

## Pré-requisitos

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [Ollama](https://ollama.com/) instalado e rodando

## Setup

```bash
# 1. Instalar dependências Python
uv sync

# 2. Baixar os modelos Ollama necessários
ollama pull nomic-embed-text   # modelo de embeddings (274 MB)
ollama pull llama3.2           # modelo de geração de texto
```

## Uso

### Indexar documentos

Coloque seus PDFs em `data/pdfs/` e rode:

```bash
uv run python src/pipeline.py index
```

Saída esperada:
```
Lendo PDFs de data/pdfs ...
  25 PDFs processados | 4110 chunks gerados
Gerando embeddings e indexando no ChromaDB ...
Concluído: 4110 chunks em 'papers'.
```

O índice é persistido em `./chroma_db/` e sobrevive entre execuções.

### Fazer perguntas

```bash
uv run python src/pipeline.py query "Como funciona o mecanismo de atenção em transformers?"
```

Saída esperada:
```
Fontes consultadas:
  [attention_paper.pdf — p. 3]
  [attention_paper.pdf — p. 7]

Resposta:
O mecanismo de atenção calcula pesos de relevância entre tokens...
```

### Baixar papers do ArXiv (opcional)

O script `download_papers.py` busca os 10 papers mais recentes de 3 categorias e os salva em `data/pdfs/`:

```bash
uv run python src/download_papers.py
```

Categorias buscadas: `LLM agents`, `RAG retrieval augmented generation`, `LLM fine-tuning`.

## Estrutura do projeto

```
fase1-rag/
├── src/
│   ├── ingestion.py        # Leitura de PDFs e chunking
│   ├── embeddings.py       # Geração de embeddings e persistência no ChromaDB
│   ├── retrieval.py        # Busca semântica de chunks relevantes
│   ├── generation.py       # Prompt engineering e geração via Ollama
│   ├── pipeline.py         # Orquestrador + CLI
│   └── download_papers.py  # Script auxiliar para baixar papers do ArXiv
├── tests/
│   ├── test_ingestion.py
│   ├── test_embeddings.py
│   ├── test_retrieval.py
│   ├── test_generation.py
│   └── test_pipeline.py
├── data/
│   ├── pdfs/               # PDFs a indexar (não commitados)
│   └── papers_index.json   # Metadados dos papers baixados
├── chroma_db/              # Índice vetorial persistente (não commitado)
├── pyproject.toml
└── NEXT_STEPS.md
```

## Desenvolvimento

```bash
# Testes (38 testes, todos mockados — sem dependência de Ollama/ChromaDB reais)
uv run pytest

# Lint e formatação
uv run ruff check . && uv run ruff format .

# Adicionar dependência
uv add <pacote>
```

## Configurações

As principais constantes estão no topo de cada módulo:

| Constante | Módulo | Valor padrão | Descrição |
|---|---|---|---|
| `CHUNK_SIZE` | `ingestion.py` | `512` | Tamanho de cada chunk em caracteres |
| `CHUNK_OVERLAP` | `ingestion.py` | `50` | Sobreposição entre chunks |
| `EMBEDDING_MODEL` | `embeddings.py` | `nomic-embed-text` | Modelo de embedding |
| `CHROMA_COLLECTION` | `embeddings.py` | `papers` | Nome da collection no ChromaDB |
| `GENERATION_MODEL` | `generation.py` | `llama3.2` | Modelo de geração |
| `TOP_K` | `retrieval.py` | `5` | Chunks retornados por query |
