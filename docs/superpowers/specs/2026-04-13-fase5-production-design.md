# Fase 5 — Produção: Design Spec

**Data:** 2026-04-13  
**Status:** Aprovado

---

## Objetivo

Transformar o pipeline RAG (Fase 1) e o agente ReAct (Fase 3) em uma API de produção com streaming, observabilidade, cache semântico e containerização.

---

## Arquitetura

### Abordagem: Adapter Layer

`fase5-production` encapsula as fases 1 e 3 via adapters. A API nunca importa as fases diretamente — só fala com os adapters. Um roteador interno decide qual adapter usar por heurística simples.

```
api.py  →  router.py  →  rag_adapter.py   →  fase1: pipeline.query()
                      →  agent_adapter.py →  fase3: agent.run()
```

As fases 1 e 3 são declaradas como `path` dependencies no `pyproject.toml`:
```toml
{path = "../fase1-rag"}
{path = "../fase3-agents"}
```

---

## Estrutura de Módulos

```
fase5-production/
├── src/
│   ├── api.py           # FastAPI app, endpoints POST /query e GET /health
│   ├── router.py        # heurística RAG vs Agent, constante AGENT_KEYWORDS
│   ├── rag_adapter.py   # encapsula fase1 pipeline.query()
│   ├── agent_adapter.py # encapsula fase3 agent.run()
│   ├── streaming.py     # gerador SSE — consome chunks Ollama token a token
│   ├── cache.py         # cache semântico com ChromaDB (collection query_cache)
│   ├── middleware.py    # logging JSON estruturado + trace_id por request
│   └── monitor.py       # amostrador de tráfego + BackgroundTask de eval
├── tests/
│   ├── test_router.py
│   ├── test_cache.py
│   ├── test_middleware.py
│   ├── test_monitor.py
│   └── test_integration.py  # requer Ollama rodando (marker: integration)
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

---

## Fluxo de Request

```
POST /query {"question": "..."}
        │
        ▼
   middleware.py
   (gera trace_id UUID, inicia timer)
        │
        ▼
    cache.py
  ┌─ hit (cosine > 0.92)? → retorna cached response
  └─ miss? → continua
        │
        ▼
   router.py
  ┌─ simples → rag_adapter.py → pipeline.query()
  └─ complexo → agent_adapter.py → agent.run()
        │
        ▼
   streaming.py
   (SSE se Accept: text/event-stream, JSON caso contrário)
        │
        ▼
   cache.py (persiste embedding + resposta, TTL 24h via metadata cached_at)
        │
        ▼
   monitor.py (BackgroundTask — fire-and-forget)
   (1-em-N requests → roda eval fase2 → loga resultado)
        │
        ▼
   middleware.py (log final JSON: trace_id, solver, cache_hit, latency_ms, question_hash)
```

---

## Módulos em Detalhe

### `router.py`

Heurística baseada em `AGENT_KEYWORDS` (lista configurável). Usa agente se a query contém mais de uma entidade + conectivo relacional ("and", "compared to", "vs", "how does X relate to Y") ou verbo de comparação explícito. RAG em todos os outros casos. Testável de forma pura — sem dependências externas.

### `cache.py`

- Collection ChromaDB separada: `query_cache`
- Mesmo modelo de embedding da Fase 1: `nomic-embed-text` via Ollama
- Threshold de similaridade cosine: `0.92` (constante `CACHE_SIMILARITY_THRESHOLD`)
- TTL: 24h — verificado via campo `cached_at` no metadata
- Interface: `get(question) -> str | None` e `set(question, answer) -> None`

### `streaming.py`

Consome o gerador de tokens do Ollama (`stream=True`) e converte para Server-Sent Events. Formato SSE: `data: <token>\n\n`, evento final `data: [DONE]\n\n`. Ativado via header `Accept: text/event-stream` — caso contrário retorna JSON `{"answer": "..."}`.

### `middleware.py`

Starlette middleware que:
1. Gera `trace_id` (8 chars hex) antes de processar
2. Injeta no request state (`request.state.trace_id`)
3. Loga ao final: `{"trace_id", "solver", "cache_hit", "latency_ms", "question_hash", "status_code"}`
- `question_hash`: SHA256 primeiros 8 chars — nunca loga o texto da query

### `monitor.py`

- Taxa de sampling configurável: `EVAL_SAMPLE_RATE = 0.1` (10% dos requests)
- Usa `random.random() < EVAL_SAMPLE_RATE` para decidir
- Enfileira `BackgroundTask` que chama o evaluator da Fase 2 com a query e resposta
- Loga resultado do eval (faithfulness, answer_relevance) com o mesmo `trace_id`

### `api.py`

```python
POST /query
  Body: {"question": str}
  Headers opcionais: Accept: text/event-stream
  Response: StreamingResponse (SSE) ou {"answer": str, "solver": str, "cached": bool}

GET /health
  Response: {"status": "ok", "ollama": bool, "chromadb": bool}
```

---

## Testes

| Arquivo | O que testa |
|---|---|
| `test_router.py` | Heurística: queries simples → RAG, queries complexas → Agent |
| `test_cache.py` | Hit acima do threshold, miss abaixo, expiração por TTL |
| `test_middleware.py` | trace_id presente no log, campos obrigatórios, question_hash não vaza texto |
| `test_monitor.py` | BackgroundTask é enfileirada na taxa correta, não bloqueia response |
| `test_integration.py` | POST /query retorna 200, /health reporta Ollama e ChromaDB (marker: integration) |

Padrão: mocks para Ollama e ChromaDB nos unitários, igual às fases anteriores.

---

## Docker

`docker-compose.yml` com dois serviços:

- **`app`**: Python 3.11-slim, copia fases 1, 3 e 5, expõe porta 8000
- **`ollama`**: imagem `ollama/ollama`, volume `ollama_data` para modelos persistirem

ChromaDB roda in-process (igual às fases anteriores), dados persistidos em volume Docker.

Healthcheck do compose: `GET /health` a cada 30s.

---

## Métricas de Sucesso

| Métrica | Alvo |
|---|---|
| Latência p95 | < 2s |
| Primeiro token SSE | < 500ms |
| Container Docker | Sobe e responde em máquina limpa |
| Cobertura de testes | ≥ 40 testes unitários + 2 integração |

---

## Dependências

```toml
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn>=0.30.0",
    "httpx>=0.27.0",        # testes de integração da API
    {path = "../fase1-rag"},
    {path = "../fase3-agents"},
]
```
