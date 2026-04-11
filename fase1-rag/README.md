# RAG do Zero — Fase 1 da Jornada de AI Engineering

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/uv-package%20manager-blueviolet)
![ruff](https://img.shields.io/badge/ruff-linter%20%2B%20formatter-orange)
![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-black)
![ChromaDB](https://img.shields.io/badge/ChromaDB-vector%20store-green)

Pipeline **RAG (Retrieval-Augmented Generation)** construído do zero, sem frameworks de abstração. Parte da minha jornada prática de AI Engineering.

```
Pergunta do usuário
       │
       ▼
┌─────────────────┐    ┌──────────────────┐    ┌────────────────┐
│ nomic-embed-text│───▶│    ChromaDB      │───▶│  Top-5 chunks  │
│ (embedding)     │    │  (vector store)  │    │  relevantes    │
└─────────────────┘    └──────────────────┘    └───────┬────────┘
                                                        │
                                           ┌────────────▼───────────┐
                                           │  System prompt + ctx   │
                                           │  + pergunta → llama3.2 │
                                           └────────────┬───────────┘
                                                        │
                                                 Resposta final
```

---

## O que é esse projeto

A maioria dos tutoriais de RAG usa LangChain ou LlamaIndex — frameworks que abstraem tudo em 3 linhas de código. Aprendi bastante com eles, mas nunca entendi de verdade o que estava acontecendo por baixo.

Esse projeto é minha resposta a isso: **implementar um pipeline RAG completo do zero**, módulo por módulo, com testes unitários, sem nenhuma camada de abstração além das bibliotecas fundamentais.

O resultado é um sistema que:
- Lê PDFs e extrai texto página por página com **PyMuPDF**
- Divide o texto em chunks com janela deslizante e overlap
- Gera embeddings localmente com **nomic-embed-text** via Ollama
- Indexa e busca vetores com **ChromaDB**
- Responde perguntas fundamentadas nos documentos com **llama3.2**
- Tudo 100% local — zero API paga, zero dado enviado para fora

---

## O que aprendi

### Chunking não é trivial
Dividir texto em pedaços é mais crítico do que parece. Tamanho do chunk (512 chars), overlap (50 chars), estratégia de divisão — cada decisão afeta diretamente a precisão da busca semântica. Um chunk muito grande vira uma "média" de muitos tópicos e perde foco. Um chunk muito pequeno perde contexto.

### Embeddings são o coração do sistema
Um embedding transforma texto em um vetor de centenas de dimensões onde textos com **significado parecido ficam próximos** — mesmo sem compartilhar nenhuma palavra. "Como fazer login?" e "Qual o processo de autenticação?" ficam próximos no espaço vetorial. É isso que torna a busca semântica possível.

### Busca semântica vs. busca por palavra-chave
Um sistema de busca por palavra-chave não encontra "mecanismo de atenção em transformers" quando a pergunta é "como LLMs focam em partes relevantes do texto?". A busca semântica sim — porque opera em cima de significado, não de tokens.

### O system prompt é a linha entre RAG e alucinação
A instrução `"Responda APENAS com base no contexto fornecido"` é o que separa um sistema confiável de um que inventa respostas plausíveis. Sem ela, o modelo complementa o contexto recuperado com memória própria — e você perde rastreabilidade e introduz alucinações silenciosas.

### Curadoria de dados importa desde o início
Ao buscar papers sobre "RAG" no ArXiv ordenado por data, recebi papers de visão computacional e robótica — o termo "RAG" é usado em outras subáreas. Sem filtros de categoria (`cat:cs.CL`), o índice fica contaminado com conteúdo irrelevante. Qualidade do dado de entrada determina qualidade da resposta.

### Mocking é a chave para testes de sistemas com dependências externas
38 testes unitários, todos passando, sem Ollama instalado, sem ChromaDB real, sem um único PDF. `unittest.mock.patch` permite testar a lógica de orquestração de forma independente das dependências — rápido, determinístico, portável.

---

## Stack e por que cada escolha

| Componente | Tecnologia | Por quê |
|---|---|---|
| **LLM local** | [Ollama](https://ollama.com/) + llama3.2 | Roda 100% offline. Zero custo por token. API HTTP local compatível com o padrão OpenAI — fácil de migrar para outro provider no futuro. |
| **Embeddings** | [Ollama](https://ollama.com/) + nomic-embed-text | Modelo especializado em representação semântica — muito mais preciso para busca do que usar o mesmo modelo de geração. Pequeno e rápido. |
| **Vector store** | [ChromaDB](https://www.trychroma.com/) | Zero configuração. Persiste em disco com SQLite + HNSW. API Python simples. Perfeito para aprender o conceito antes de ir para soluções de produção. |
| **Leitura de PDF** | [PyMuPDF](https://pymupdf.readthedocs.io/) | Extração de texto mais precisa que pdfplumber e pypdf, especialmente em PDFs técnicos com layouts complexos. Mantém metadados de página. |
| **Gerenciamento de ambiente** | [uv](https://docs.astral.sh/uv/) | Resolve dependências em < 1 segundo. `pyproject.toml` como única fonte de verdade. O pip/venv de 2024. |
| **Lint e formatação** | [ruff](https://docs.astral.sh/ruff/) | Substitui flake8 + isort + black em um único binário Rust. Extremamente rápido. |

---

## Como rodar

### Pré-requisitos

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [Ollama](https://ollama.com/) instalado e rodando

### Setup

```bash
# Clonar o repositório
git clone https://github.com/LeoBR01/ai-engineering-journey.git
cd ai-engineering-journey/fase1-rag

# Instalar dependências Python
uv sync

# Baixar os modelos Ollama
ollama pull nomic-embed-text   # modelo de embeddings (274 MB)
ollama pull llama3.2           # modelo de geração de texto
```

### Indexar documentos

```bash
# Opção 1: baixar papers do ArXiv automaticamente
uv run python src/download_papers.py
# → salva ~25 PDFs em data/pdfs/ e metadados em data/papers_index.json

# Opção 2: copiar seus próprios PDFs
cp meu_documento.pdf data/pdfs/

# Indexar tudo no ChromaDB
uv run python src/pipeline.py index
```

Saída esperada:
```
Lendo PDFs de data/pdfs ...
  25 PDFs processados | 4110 chunks gerados
Gerando embeddings e indexando no ChromaDB ...
Concluído: 4110 chunks em 'papers'.
```

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

### Testes e qualidade de código

```bash
# Rodar os 38 testes unitários (não requer Ollama nem ChromaDB)
uv run pytest

# Lint e formatação
uv run ruff check . && uv run ruff format .
```

---

## Estrutura do projeto

```
fase1-rag/
├── src/
│   ├── ingestion.py        # Leitura de PDFs e chunking com overlap
│   ├── embeddings.py       # Embeddings via Ollama + persistência ChromaDB
│   ├── retrieval.py        # Busca semântica top-K + formatação de contexto
│   ├── generation.py       # Prompt RAG + geração via Ollama
│   ├── pipeline.py         # Orquestrador + CLI (index / query)
│   └── download_papers.py  # Coleta automática de papers do ArXiv
├── tests/                  # 38 testes unitários, todos mockados
├── data/
│   └── papers_index.json   # Metadados dos papers baixados
├── pyproject.toml
├── NEXT_STEPS.md
└── README.md
```

---

## Próximos passos — Fase 2: Evals

A Fase 1 entrega um pipeline que *funciona*. A Fase 2 vai responder: **funciona bem?**

Sem métricas, é impossível saber se uma mudança de configuração melhorou ou piorou o sistema. Os planos para a Fase 2:

- **Dataset de avaliação:** pares `(pergunta, resposta_esperada)` sobre os papers indexados
- **Métricas de retrieval:** Recall@K e MRR — o chunk certo está entre os K retornados?
- **Métricas de geração:** faithfulness e relevância via LLM-as-a-judge
- **Experimentos de chunking:** comparar 256 / 512 / 1024 chars e chunking semântico
- **Re-indexação idempotente:** chunk_id determinístico baseado em hash(source + page + índice)
- **Filtros de categoria no ArXiv:** garantir papers relevantes para NLP/IA (`cat:cs.CL`)

> Repositório completo da jornada: [LeoBR01/ai-engineering-journey](https://github.com/LeoBR01/ai-engineering-journey)
