# Jornada AI Engineering — Status e Próximos Passos

## Status das Fases

| Fase | Título | Status |
|---|---|---|
| Fase 1 | Pipeline RAG | ✅ Completo |
| Fase 2 | Sistema de Evals | ✅ Completo |
| Fase 3 | Agente ReAct | ✅ Completo |
| Fase 4 | Fine-tuning | 🔜 Próxima |

---

## ✅ Fase 1 — Pipeline RAG (completo)

Pipeline RAG 100% local com ChromaDB + Ollama.

| Módulo | Função |
|---|---|
| `ingestion.py` | Leitura de PDFs e chunking |
| `embeddings.py` | Geração de embeddings e persistência no ChromaDB |
| `retrieval.py` | Busca semântica por similaridade cosine |
| `generation.py` | Geração de resposta via Ollama com contexto |
| `pipeline.py` | Orquestração do fluxo completo |

**Stack:** Python, ChromaDB, Ollama (llama3.2 + nomic-embed-text), ~4.110 chunks de papers do ArXiv.

---

## ✅ Fase 2 — Sistema de Evals (completo)

Sistema de avaliação quantitativa para o pipeline RAG da Fase 1.

| Módulo | Função |
|---|---|
| `dataset.py` | Dataset de 15 perguntas (easy/medium/hard) |
| `metrics_retrieval.py` | Recall@K e MRR |
| `metrics_generation.py` | Faithfulness e Answer Relevance via LLM-as-judge |
| `evaluator.py` | Orquestrador com injeção de dependência |
| `report.py` | Persiste resultados com timestamp |
| `run_eval.py` | Integração com ChromaDB e Ollama da Fase 1 |

**Resultados (2026-04-11):** Faithfulness médio 0.57 · Answer Relevance médio 0.60 · 44 testes unitários.

---

## ✅ Fase 3 — Agente ReAct (completo)

Agente ReAct implementado do zero, sem LangChain ou frameworks externos.
Usa o pipeline RAG da Fase 1 como ferramenta de busca para perguntas multi-hop.

| Módulo | Função |
|---|---|
| `tools.py` | `search_papers`, `calculate` (AST seguro), `summarize` |
| `prompt.py` | `SYSTEM_PROMPT` + `build_react_prompt` com histórico |
| `agent.py` | `ReActAgent`, `AgentResult`, `Step`, `_parse_llm_output` |
| `run_agent.py` | CLI `--question` / modo interativo |

**Destaques:**
- `calculate` usa `ast.parse` + caminhamento da AST — rejeita `Call`, `Import`, atributos
- `SYSTEM_PROMPT` enviado no campo `system` do `ollama.chat` para maior autoridade
- Gatilho explícito de busca obrigatória no passo 1 (elimina alucinação sem retrieval)
- Loop tolera saída malformada do LLM e continua sem travar
- 30 testes unitários, todos mockados, todos passando

**Resultado dos testes reais (2026-04-12):** ciclo Thought → Action: search_papers → Observation → Final Answer
confirmado nas duas perguntas de teste. Qualidade da resposta limitada pelo corpus, não pelo agente.

---

## 🔜 Fase 4 — Fine-tuning

### Objetivo

Fazer fine-tuning de um modelo de linguagem pequeno para melhorar a qualidade das respostas
do pipeline RAG construído nas fases anteriores. Comparar o modelo base vs. o modelo
fine-tunado usando o sistema de evals da Fase 2.

### O que explorar

- **Dataset de fine-tuning:** gerar pares (pergunta, resposta ideal) a partir dos papers do
  ArXiv já indexados — o corpus já existe, só falta curar exemplos de qualidade
- **Técnica:** LoRA / QLoRA para fine-tuning eficiente em GPU consumer (4-8GB VRAM)
- **Modelo base candidato:** Llama 3.2 (3B) ou Mistral 7B — compatíveis com Ollama
- **Framework:** Unsloth ou TRL (HuggingFace) para o treino; Ollama para servir o modelo
  fine-tunado localmente

### Métricas de sucesso

- Faithfulness médio > 0.70 (vs. 0.57 do baseline da Fase 2)
- Answer Relevance médio > 0.70 (vs. 0.60 do baseline)
- O modelo fine-tunado segue o formato ReAct mais consistentemente que o base

### Arquitetura planejada

```
fase4-finetuning/
├── data/
│   ├── raw/              # pares (pergunta, resposta) brutos
│   └── processed/        # dataset no formato de treino (Alpaca / ChatML)
├── src/
│   ├── generate_dataset.py  # usa o pipeline RAG para gerar exemplos
│   ├── train.py             # loop de fine-tuning com LoRA
│   ├── export.py            # exporta o adaptador para GGUF / Ollama
│   └── evaluate.py          # roda os evals da Fase 2 no modelo fine-tunado
└── tests/
    └── test_dataset.py
```

### Dependências

- `unsloth` ou `trl` para fine-tuning
- `datasets` (HuggingFace) para manipulação do dataset
- GPU com ≥ 4GB VRAM (ou Google Colab T4) para o treino
- Modelo base em formato GGUF para Ollama
