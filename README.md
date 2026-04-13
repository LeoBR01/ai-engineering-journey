# AI Engineering Journey

> Aprendizado prático e progressivo de engenharia de IA — construindo do zero, sem frameworks que escondem o que importa.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/uv-gerenciador-DE5FE9?logo=python&logoColor=white)
![ruff](https://img.shields.io/badge/ruff-lint%20%2B%20format-D7FF64?logo=python&logoColor=black)
![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-black?logo=ollama&logoColor=white)
![Testes](https://img.shields.io/badge/testes-158%20unitários-brightgreen)

---

## O que é esse projeto

Uma jornada de aprendizado estruturada em fases, onde cada fase constrói sobre a anterior.

A ideia central é simples: **entender o que está por baixo antes de usar a abstração**. Em vez de importar uma linha de LangChain e seguir em frente, cada componente é implementado do zero — com testes, com documentação, com as perguntas "por que isso funciona assim?" respondidas no código.

O projeto é 100% local. Sem chamadas para APIs pagas. LLMs rodando via Ollama na sua própria máquina.

---

## Visão geral das fases

```
Fase 1 ──────► Fase 2 ──────► Fase 3 ──────► Fase 4 ──────► Fase 5
  RAG            Evals         Agentes        Fine-tuning    Produção
  do zero        quant.        ReAct          QLoRA local    e API
  ✅ Feita       ✅ Feita       ✅ Feita        ✅ Feita        🔮 Futura
```

| Fase | Tema | Status | Descrição curta |
|------|------|--------|-----------------|
| [Fase 1](#fase-1--pipeline-rag-do-zero) | Pipeline RAG | ✅ Completa | Ingestion → Embeddings → Retrieval → Geração |
| [Fase 2](#fase-2--avaliação-quantitativa) | Avaliação quantitativa | ✅ Completa | Métricas de retrieval e geração, LLM-as-judge |
| [Fase 3](#fase-3--agentes-react) | Agentes ReAct | ✅ Completa | Agente do zero com ferramentas, loop ReAct, parse regex |
| [Fase 4](#fase-4--fine-tuning-local) | Fine-tuning local | ✅ Completa | QLoRA para melhorar Faithfulness, Answer Relevance e formato ReAct |
| [Fase 5](#fase-5--produção-futura) | Produção | 🔮 Futura | API, observabilidade, deploy e monitoramento |

---

## Pré-requisitos globais

```bash
# Instalar uv (gerenciador de pacotes Python)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Instalar Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Baixar os modelos usados
ollama pull llama3.2           # LLM de geração (~2 GB)
ollama pull nomic-embed-text   # Modelo de embeddings (~274 MB)
```

Cada fase tem seu próprio ambiente virtual gerenciado pelo `uv`. Não há dependências compartilhadas entre fases — cada uma é autossuficiente.

---

## Fase 1 — Pipeline RAG do zero

**Diretório:** `fase1-rag/`
**Documentação:** [`fase1-rag/README.md`](fase1-rag/README.md)
**Testes:** 38 unitários, todos mockados

### O que foi construído

Um pipeline RAG completo implementado com bibliotecas fundamentais — sem LangChain, sem LlamaIndex. Cada componente tem responsabilidade única e pode ser entendido de forma isolada.

```
PDF → [Ingestion] → chunks → [Embeddings] → vetores → [ChromaDB]
                                                            ↑
Pergunta → embedding → busca semântica → top-K chunks ──────┘
                                                     ↓
                                              [Generation]
                                                     ↓
                                              Resposta fundamentada
```

### Módulos

| Módulo | Responsabilidade |
|--------|-----------------|
| `src/ingestion.py` | Leitura de PDFs com PyMuPDF, chunking com overlap (512 chars, overlap 50) |
| `src/embeddings.py` | Geração via `nomic-embed-text`, persistência em ChromaDB em lotes de 500 |
| `src/retrieval.py` | Busca semântica top-K por cosine similarity, formatação do contexto |
| `src/generation.py` | Montagem do prompt RAG, geração via `llama3.2` |
| `src/pipeline.py` | Orquestra os 4 módulos, expõe CLI `index` e `query` |
| `src/download_papers.py` | Baixa papers do ArXiv por tema e afiliação (Anthropic, DeepMind, OpenAI…) |

### Stack

| Componente | Tecnologia | Por quê |
|------------|-----------|---------|
| LLM local | Ollama + llama3.2 | 100% offline, zero custo, API compatível |
| Embeddings | nomic-embed-text | Especializado em semântica, 274 MB |
| Vector store | ChromaDB | Zero config, persistência SQLite + HNSW |
| PDF | PyMuPDF | Extração precisa de texto em papers técnicos |

### Como rodar

```bash
cd fase1-rag
uv sync

# Baixar papers do ArXiv (gera ~25 PDFs em data/pdfs/)
uv run python src/download_papers.py

# Indexar os PDFs no ChromaDB
uv run python src/pipeline.py index

# Fazer uma pergunta
uv run python src/pipeline.py query "Como o mecanismo de atenção funciona nos transformers?"

# Rodar testes
uv run pytest

# Lint e formatação
uv run ruff check . && uv run ruff format .
```

### O que foi aprendido

- **Chunking não é trivial.** Tamanho (512 chars) e overlap (50 chars) afetam diretamente a qualidade. Um chunk grande demais vira "média" de tópicos; pequeno demais perde contexto.
- **Embeddings são o coração do RAG.** Transformam texto em vetores onde *significados próximos ficam próximos*, independente das palavras usadas.
- **Busca semântica > busca por palavra-chave.** "Como LLMs focam em partes relevantes?" encontra chunks sobre mecanismo de atenção mesmo sem usar a palavra "atenção".
- **System prompt é a diferença entre alucinação e resposta fundamentada.** "Responda APENAS com base no contexto fornecido" elimina invenções silenciosas.
- **Mocking torna testes rápidos e portáveis.** 38 testes que rodam em segundos, sem precisar de Ollama ou ChromaDB reais.

---

## Fase 2 — Avaliação quantitativa

**Diretório:** `fase2-evals/`
**Testes:** 44 unitários + 4 de integração

### O que foi construído

A Fase 1 entrega um pipeline que *funciona*. A Fase 2 responde: **funciona bem?**

Um sistema de avaliação com dataset de 15 perguntas (easy/medium/hard), métricas de retrieval e geração, e um experimento real com os resultados documentados.

### Módulos

| Módulo | Responsabilidade |
|--------|-----------------|
| `src/dataset.py` | Carregamento, validação e persistência do dataset de evals (TypedDict) |
| `src/metrics_retrieval.py` | Recall@K e MRR para medir qualidade do retrieval |
| `src/metrics_generation.py` | Faithfulness e Answer Relevance via LLM-as-judge |
| `src/evaluator.py` | Orquestra a avaliação completa com injeção de dependência |
| `src/report.py` | Formata e persiste relatórios em JSON e texto |
| `src/run_eval.py` | Integra com a Fase 1, CLI, exibe tabela colorida com Rich |

### Métricas

**Retrieval:**
- **Recall@K** — que % dos chunks relevantes aparecem nos top-K resultados
- **MRR (Mean Reciprocal Rank)** — quão cedo o primeiro resultado relevante aparece

**Geração (LLM-as-judge com `llama3.2`):**
- **Faithfulness** — a resposta está suportada pelo contexto recuperado? (0.0–1.0)
- **Answer Relevance** — a resposta de fato responde à pergunta? (0.0–1.0)

### Como rodar

```bash
cd fase2-evals
uv sync

# Rodar avaliação completa (requer ChromaDB da Fase 1 populado)
uv run eval-run

# Rodar testes
uv run pytest

# Apenas testes de integração (requerem Ollama + ChromaDB)
uv run pytest -m integration
```

### Resultados do primeiro experimento (2026-04-11)

Configuração: `llama3.2` como gerador e juiz, `nomic-embed-text` para embeddings, 4.110 chunks de 25 papers do ArXiv.

| Métrica | Resultado | Interpretação |
|---------|-----------|---------------|
| Recall@5 | 0.00 | Esperado — `relevant_chunk_ids` não anotados manualmente |
| MRR | 0.00 | Mesma razão |
| Faithfulness | **0.57** | Médio — respostas parcialmente suportadas pelo contexto |
| Answer Relevance | **0.60** | Médio — respostas respondem parte das perguntas |

**Descoberta interessante:** Perguntas *hard* obtiveram scores melhores que *easy*. Motivo: papers técnicos do ArXiv discutem limitações e trade-offs (conteúdo hard) mas raramente definem termos básicos como "chunking" (conteúdo easy).

### O que foi aprendido

- **LLM-as-judge funciona** — mas usar o mesmo modelo para gerar e avaliar cria viés. Fase 3 pode explorar juízes externos.
- **Recall@K exige anotação humana** — sem marcar quais chunks são relevantes para cada pergunta, a métrica não tem sinal.
- **Curadoria do dataset importa mais que o tamanho** — 15 perguntas bem construídas revelam mais que 100 perguntas genéricas.
- **Injeção de dependência simplifica testes** — passar `retrieve_fn` e `generate_fn` como parâmetros elimina acoplamento com Ollama/ChromaDB nos testes unitários.

---

## Fase 3 — Agentes ReAct

**Diretório:** `fase3-agents/`
**Testes:** 30 unitários, todos mockados

### O que foi construído

Um agente ReAct implementado do zero, sem frameworks externos. O agente usa o RAG como *ferramenta*, decidindo quando buscar, calcular ou resumir — em vez de seguir sempre os mesmos passos.

```
Pergunta
   ↓
[Thought] — o que eu preciso fazer?
   ↓
[Action] — qual ferramenta usar? com quais argumentos?
   ↓
[Observation] — resultado da ferramenta
   ↓
[Thought] — tenho o suficiente para responder?
   ↓ (se não, volta para Action)
[Final Answer] — resposta fundamentada
```

### Módulos

| Módulo | Responsabilidade |
|--------|-----------------|
| `src/tools.py` | `search_papers` (ChromaDB), `calculate` (AST seguro), `summarize` (Ollama) |
| `src/prompt.py` | `SYSTEM_PROMPT` no papel `system`, `build_react_prompt` com histórico |
| `src/agent.py` | `ReActAgent`, loop ReAct, parse via regex, `AgentResult`, `Step` |
| `src/run_agent.py` | CLI `--question` e modo interativo |

### Stack

- Python puro + Ollama (`llama3.2`) — sem frameworks de agentes
- ChromaDB da Fase 1 reaproveitado como tool de busca semântica
- Parse via regex: `Thought`, `Action`, `Action Input`, `Final Answer`

### Como rodar

```bash
cd fase3-agents
uv sync

# Pergunta única
uv run python src/run_agent.py --question "Qual métrica o paper X usa para avaliar agentes?"

# Modo interativo
uv run python src/run_agent.py

# Testes
uv run pytest
```

### O que foi aprendido

- **SYSTEM_PROMPT no papel `system` é crítico.** Colocar as instruções ReAct como mensagem de sistema do Ollama — e não como `user` — faz o modelo respeitar o formato com muito mais consistência.
- **Parse frágil revela limitações do modelo base.** O llama3.2 (3B) segue o formato em ~70% dos casos; modelos maiores seriam muito mais consistentes. Isso motivou a Fase 4.
- **ReAct sem memória entre sessões.** O histórico de raciocínio existe apenas durante uma execução — agentes persistentes requerem serialização.
- **Ferramentas simples bastam.** `search_papers` + `calculate` + `summarize` cobrem a maioria das perguntas sobre os papers. Adicionar ferramentas demais aumenta a confusão do modelo.

---

## Fase 4 — Fine-tuning local

**Diretório:** `fase4-finetuning/`
**Testes:** 46 unitários + 2 de integração

### O que foi construído

Fine-tuning supervisionado (SFT) com QLoRA no llama3.2 3B para atacar os dois gargalos medidos na Fase 2 e o problema de formato da Fase 3. Dataset sintético gerado a partir do ChromaDB existente.

```
ChromaDB (4.110 chunks)
    ↓
generate_dataset.py → data/raw/ → data/processed/train.jsonl
    ↓
train.py (QLoRA, r=16, lora_alpha=32) → output/lora_adapter/
    ↓
export.py → output/model_gguf/ → Ollama (llama3.2-finetuned)
    ↓
evaluate.py → relatório comparativo (baseline vs. fine-tuned)
```

### Módulos

| Módulo | Responsabilidade |
|--------|-----------------|
| `src/generate_dataset.py` | Geração de pares RAG (chunk→pergunta+resposta) e ciclos ReAct sintéticos perfeitos |
| `src/train.py` | `TrainConfig` + loop QLoRA com Unsloth/TRL SFTTrainer |
| `src/export.py` | Adapter LoRA → GGUF (Q4_K_M) → `ollama create` |
| `src/evaluate.py` | Reutiliza evals da Fase 2, gera relatório baseline vs. fine-tuned |

### Stack

| Componente | Tecnologia | Por quê |
|------------|-----------|---------|
| Fine-tuning | Unsloth + TRL SFTTrainer | 2x mais rápido, 30% menos VRAM que TRL puro |
| Quantização | QLoRA 4-bit (NF4) | ~6GB VRAM — viável em GPU consumer |
| Dataset | ChatML (messages[]) | Formato nativo do llama3.2 Instruct |
| Export | GGUF Q4_K_M via Unsloth | Compatível com Ollama local |

### Como rodar (requer GPU ≥8GB VRAM)

```bash
cd fase4-finetuning
uv sync

# Instalar Unsloth + TRL (string varia por CUDA version)
uv run pip install "unsloth[cu124-torch250]" trl peft

# 1. Gerar dataset
uv run generate-dataset --rag-output data/raw/rag_pairs.jsonl \
                        --react-output data/raw/react_cycles.jsonl \
                        --processed-output data/processed/train.jsonl

# 2. Treinar
uv run train-model --dataset data/processed/train.jsonl --output output/lora_adapter

# 3. Exportar para Ollama
uv run export-model --adapter output/lora_adapter --gguf-output output/model_gguf

# 4. Avaliar (comparar com baseline da Fase 2)
uv run evaluate-model --model llama3.2-finetuned --output data/results

# Testes unitários (sem GPU)
uv run pytest --ignore=tests/test_integration.py
```

### Métricas-alvo

| Métrica | Baseline (Fase 2) | Meta |
|---------|-------------------|------|
| Faithfulness | 0.57 | > 0.70 |
| Answer Relevance | 0.60 | > 0.70 |
| Formato ReAct correto | ~70% | > 90% |

### O que foi aprendido

- **Ciclos ReAct sintéticos > capturar saídas do modelo quebrado.** Construir os exemplos de treino programaticamente — com observações reais do ChromaDB e Final Answers gerados pelo LLM — garante formato perfeito em 100% dos casos.
- **Module-level try/except habilita testes sem GPU.** Importar Unsloth/TRL no topo do módulo com fallback `= None` permite `@patch("src.train.FastLanguageModel")` funcionar em qualquer máquina.
- **lora_alpha = 2×r é o padrão do paper original.** Com r=16 e lora_alpha=32, o scaling factor é 2.0 — calibração padrão da literatura LoRA.

---

## Fase 5 — Produção *(futura)*

**Diretório:** `fase5-production/` *(a criar)*

### O que será construído

Transformar o pipeline da Fase 1 (e o agente da Fase 3) em uma API com observabilidade, rate limiting, e capacidade de monitorar qualidade em produção.

### Tópicos planejados

- **API REST** com FastAPI — expor o pipeline RAG como serviço
- **Streaming de respostas** — token por token via Server-Sent Events
- **Observabilidade** — logging estruturado, tracing de requests, métricas de latência
- **Evals em produção** — amostrar respostas reais e avaliar com o sistema da Fase 2
- **Cache semântico** — evitar re-computar embeddings para queries similares
- **Containerização** — Dockerfile para reproducibilidade

---

## Estrutura do repositório

```
ai-engineering-journey/
│
├── fase1-rag/                    # ✅ Pipeline RAG do zero
│   ├── src/                      # 5 módulos (ingestion, embeddings, retrieval, generation, pipeline)
│   ├── tests/                    # 38 testes unitários
│   ├── data/
│   │   ├── pdfs/                 # Papers ArXiv (não versionados)
│   │   ├── papers_index.json     # Metadados dos papers
│   │   └── chroma_db/            # Vector store (não versionado)
│   ├── pyproject.toml
│   ├── README.md
│   └── CLAUDE.md
│
├── fase2-evals/                  # ✅ Sistema de avaliação quantitativa
│   ├── src/                      # 6 módulos
│   ├── tests/                    # 44 testes (40 unitários + 4 integração)
│   ├── data/
│   │   ├── eval_dataset.json     # 15 perguntas anotadas
│   │   └── results/              # Resultados de experimentos com timestamp
│   ├── pyproject.toml
│   └── CLAUDE.md
│
├── fase3-agents/                 # ✅ Agente ReAct do zero
│   ├── src/                      # 4 módulos (tools, prompt, agent, run_agent)
│   ├── tests/                    # 30 testes unitários
│   ├── pyproject.toml
│   └── CLAUDE.md
│
├── fase4-finetuning/             # ✅ Fine-tuning QLoRA
│   ├── src/                      # 4 módulos (generate_dataset, train, export, evaluate)
│   ├── tests/                    # 46 testes (44 unitários + 2 integração)
│   ├── data/                     # raw/, processed/, results/
│   ├── pyproject.toml
│   └── CLAUDE.md
│
├── fase5-production/             # 🔮 Futura
│
├── docs/
│   ├── NEXT_STEPS.md
│   └── superpowers/              # specs e planos de implementação
│
├── .gitignore
└── README.md                     # Este arquivo
```

---

## Princípios do projeto

**Construir antes de abstrair.** Cada componente é implementado manualmente antes de considerar bibliotecas de alto nível. Isso garante que, quando a abstração for usada, o que ela faz debaixo seja compreendido.

**Testar tudo.** Todo módulo tem testes antes de ir para produção. Mocking extensivo garante testes rápidos sem dependências externas.

**Documentar o raciocínio.** O *por quê* de cada escolha de design fica no código e na documentação — não apenas o *o quê*.

**Medir para melhorar.** Sem métricas, "funciona" não significa nada. A Fase 2 existe para que todas as fases seguintes possam ser comparadas contra uma baseline objetiva.

**Progredir incrementalmente.** Cada fase constrói sobre a anterior. A Fase 3 reusa o ChromaDB da Fase 1 e as métricas da Fase 2. A Fase 5 coloca em produção o que as fases anteriores construíram.

---

## Convenções

- **Gerenciamento de pacotes:** `uv` em todas as fases. Nunca `pip install` diretamente.
- **Lint e formatação:** `ruff check . && ruff format .` antes de qualquer commit.
- **Testes:** `pytest` com mocks para dependências externas (Ollama, ChromaDB).
- **Idioma:** Comentários e documentação em português. Código e nomes de variáveis em inglês.
- **Commits:** Conventional Commits (`feat:`, `fix:`, `chore:`, `docs:`).

---

## Progresso

- [x] Fase 1 — Pipeline RAG completo com 38 testes
- [x] Fase 2 — Avaliação quantitativa com experimento real rodado
- [x] Fase 3 — Agente ReAct do zero com 30 testes
- [x] Fase 4 — Fine-tuning QLoRA com 46 testes
- [ ] Fase 5 — API, observabilidade e produção
