# Jornada AI Engineering — Status e Próximos Passos

## Status das Fases

| Fase | Título | Status |
|---|---|---|
| Fase 1 | Pipeline RAG | ✅ Completo |
| Fase 2 | Sistema de Evals | ✅ Completo |
| Fase 3 | Agente ReAct | ✅ Completo |
| Fase 4 | Fine-tuning QLoRA | ✅ Completo |
| Fase 5 | Produção | ✅ Completo |

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
- Loop tolera saída malformada do LLM e continua sem travar
- 30 testes unitários, todos mockados, todos passando

---

## ✅ Fase 4 — Fine-tuning QLoRA (completo)

Fine-tuning supervisionado (SFT) com QLoRA no llama3.2 3B para melhorar
qualidade RAG e aderência ao formato ReAct.

| Módulo | Função |
|---|---|
| `generate_dataset.py` | Pares RAG sintéticos + ciclos ReAct programáticos |
| `train.py` | QLoRA com Unsloth/TRL, `TrainConfig`, `lora_alpha=32` |
| `export.py` | Adapter → GGUF Q4_K_M → `ollama create` |
| `evaluate.py` | Reutiliza evals da Fase 2, relatório baseline vs. fine-tuned |

**Métricas-alvo:** Faithfulness > 0.70 · Answer Relevance > 0.70 · Formato ReAct > 90%

**Stack:** Unsloth + TRL SFTTrainer, QLoRA 4-bit (NF4), ChatML, GGUF via Ollama.
46 testes unitários + 2 integração.

---

## ✅ Fase 5 — Produção (completo)

API FastAPI de produção expondo o pipeline RAG (Fase 1) e o agente ReAct (Fase 3).

| Módulo | Função |
|---|---|
| `router.py` | Heurística RAG vs Agent baseada em AGENT_KEYWORDS |
| `cache.py` | Cache semântico ChromaDB, similaridade cosine ≥ 0.92, TTL 24h |
| `rag_adapter.py` | Encapsula pipeline RAG da Fase 1 via sys.modules isolation |
| `agent_adapter.py` | Encapsula agente ReAct da Fase 3 via sys.modules isolation |
| `streaming.py` | SSE token-a-token via Ollama streaming |
| `middleware.py` | Logging JSON estruturado com trace_id por request |
| `monitor.py` | BackgroundTask para evals em 10% do tráfego |
| `api.py` | FastAPI app — POST /query (JSON ou SSE), GET /health |

**Stack:** FastAPI, uvicorn, ChromaDB, Ollama, Docker + docker-compose.
47 testes unitários + 4 integração.
