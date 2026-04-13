# Fase 5 — Produção Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expor o pipeline RAG (Fase 1) e o agente ReAct (Fase 3) como uma API FastAPI de produção com streaming SSE, cache semântico, logging estruturado, evals em background e containerização Docker.

**Architecture:** Adapter layer — `RagAdapter` e `AgentAdapter` encapsulam as fases 1 e 3 via sys.path. Um `router.py` por heurística decide qual adapter usar. Middleware injeta `trace_id` e loga em JSON. Cache semântico usa ChromaDB. Evals usam `BackgroundTasks` do FastAPI (fire-and-forget).

**Tech Stack:** FastAPI 0.115+, uvicorn, chromadb, ollama Python SDK, httpx (testes), pytest, ruff, Docker + docker-compose.

---

## Interfaces das fases dependentes (lidas do código)

```python
# fase1-rag/src/retrieval.py
def retrieve(query: str, collection=None, top_k: int = 5) -> list[dict]: ...
def format_context(chunks: list[dict]) -> str: ...

# fase1-rag/src/generation.py
GENERATION_MODEL = "llama3.2"
def generate(query: str, chunks: list[dict], model: str = ...) -> str: ...
def build_prompt(context: str, question: str) -> list[dict]: ...
# response usa atributo: response.message.content  (não dict)
# streaming: chunk.message.content

# fase3-agents/src/agent.py
@dataclass
class AgentResult:
    answer: str
    steps: list[Step]
    truncated: bool
    total_steps: int

class ReActAgent:
    def __init__(self, tools: dict, model: str = "llama3.2", max_steps: int = 10): ...
    def run(self, question: str) -> AgentResult: ...

# fase3-agents/src/tools.py
# search_papers, calculate, summarize

# fase2-evals/src/metrics_generation.py
def answer_relevance(answer: str, question: str, model: str = "llama3.2") -> float: ...
def faithfulness(answer: str, context: str, model: str = "llama3.2") -> float: ...
```

---

## Mapa de arquivos

| Arquivo | Responsabilidade |
|---|---|
| `src/router.py` | Heurística RAG vs Agent via AGENT_KEYWORDS |
| `src/cache.py` | SemanticCache: ChromaDB + embedding da query, threshold 0.92, TTL 24h |
| `src/rag_adapter.py` | Encapsula fase1: retrieve + generate (+ stream) |
| `src/agent_adapter.py` | Encapsula fase3: ReActAgent com TOOLS |
| `src/streaming.py` | Formatação SSE: `to_sse(Iterator)` e `string_to_sse(str)` |
| `src/middleware.py` | LoggingMiddleware: trace_id, JSON log, latência |
| `src/monitor.py` | maybe_schedule_eval: sampling 10% + BackgroundTask |
| `src/api.py` | FastAPI app, lifespan, /query, /health |
| `tests/test_router.py` | Casos RAG vs Agent por heurística |
| `tests/test_cache.py` | Hit/miss/expirado, sem ChromaDB/Ollama reais |
| `tests/test_rag_adapter.py` | Mocks de retrieve, generate, ollama.chat |
| `tests/test_agent_adapter.py` | Mock de ReActAgent |
| `tests/test_streaming.py` | Formato SSE puro |
| `tests/test_middleware.py` | TestClient + caplog |
| `tests/test_monitor.py` | Sampling, BackgroundTask, log de métricas |
| `tests/test_api.py` | TestClient com globals mockados |
| `tests/test_integration.py` | Requer Ollama + ChromaDB reais (marker: integration) |
| `Dockerfile` | Build context = raiz da ai-engineering-journey |
| `docker-compose.yml` | Serviços: app + ollama |

---

## Task 1: Scaffold do projeto

**Files:**
- Create: `fase5-production/pyproject.toml`
- Create: `fase5-production/src/__init__.py`
- Create: `fase5-production/tests/__init__.py`
- Create: `fase5-production/data/.gitkeep`

- [ ] **Step 1: Criar estrutura de diretórios**

```bash
cd /home/lbr/projetos/ai-engineering-journey
mkdir -p fase5-production/src fase5-production/tests fase5-production/data
touch fase5-production/src/__init__.py fase5-production/tests/__init__.py fase5-production/data/.gitkeep
```

- [ ] **Step 2: Criar pyproject.toml**

```toml
# fase5-production/pyproject.toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src"]

[project]
name = "fase5-production"
version = "0.1.0"
description = "API de produção: RAG pipeline + Agente ReAct via FastAPI"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.30.0",
    "chromadb>=0.5.0",
    "ollama>=0.3.0",
    "httpx>=0.27.0",
    "pytest>=8.0.0",
    "pytest-cov>=5.0.0",
]

[dependency-groups]
dev = [
    "ruff>=0.15.0",
]

[tool.ruff]
line-length = 88
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]
ignore = []

[tool.ruff.format]
quote-style = "double"
indent-style = "space"

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["integration: requer Ollama e ChromaDB rodando"]
```

- [ ] **Step 3: Instalar dependências**

```bash
cd fase5-production && uv sync
```

Expected: `.venv` criado, `fastapi`, `uvicorn`, `chromadb`, `ollama`, `httpx`, `pytest` instalados.

- [ ] **Step 4: Commit**

```bash
git add fase5-production/
git commit -m "feat: scaffold fase5-production"
```

---

## Task 2: router.py

**Files:**
- Create: `fase5-production/src/router.py`
- Create: `fase5-production/tests/test_router.py`

- [ ] **Step 1: Escrever o teste**

```python
# fase5-production/tests/test_router.py
import pytest

from src.router import route


@pytest.mark.parametrize(
    "question,expected",
    [
        ("What is RAG?", "rag"),
        ("How does attention work?", "rag"),
        ("Explain transformers", "rag"),
        ("", "rag"),
        ("Compare RAG and fine-tuning", "agent"),
        ("RAG vs fine-tuning: which is better?", "agent"),
        ("What is the difference between RAG and RLHF?", "agent"),
        ("How does RAG compare to fine-tuning in practice?", "agent"),
        ("Explain both RAG and fine-tuning and their trade-offs", "agent"),
        ("What is the relationship between attention and transformers?", "agent"),
    ],
)
def test_route(question: str, expected: str) -> None:
    assert route(question) == expected
```

- [ ] **Step 2: Rodar para ver FAIL**

```bash
cd fase5-production && uv run pytest tests/test_router.py -v
```

Expected: `ImportError: cannot import name 'route' from 'src.router'`

- [ ] **Step 3: Implementar router.py**

```python
# fase5-production/src/router.py
"""Roteador — decide se a query vai para RAG ou para o Agente ReAct.

Usa heurística baseada em palavras-chave para identificar queries complexas
(comparação, relação entre conceitos) que se beneficiam do agente ReAct.
Queries simples vão direto para o pipeline RAG.
"""

import re

# Palavras-chave que indicam query complexa (multi-hop ou comparação)
AGENT_KEYWORDS: list[str] = [
    r"\bcompare\b",
    r"\bcompared to\b",
    r"\bvs\.?\b",
    r"\bversus\b",
    r"\bdifference between\b",
    r"\bhow does .+ compare\b",
    r"\brelationship between\b",
    r"\bboth\b.+\band\b",
]

_PATTERNS = [re.compile(kw, re.IGNORECASE) for kw in AGENT_KEYWORDS]


def route(question: str) -> str:
    """Retorna 'agent' se a query é complexa, 'rag' caso contrário.

    Args:
        question: pergunta do usuário.

    Returns:
        'agent' ou 'rag'.
    """
    for pattern in _PATTERNS:
        if pattern.search(question):
            return "agent"
    return "rag"
```

- [ ] **Step 4: Rodar para ver PASS**

```bash
uv run pytest tests/test_router.py -v
```

Expected: 10 passed.

- [ ] **Step 5: Lint**

```bash
uv run ruff check src/router.py tests/test_router.py && uv run ruff format src/router.py tests/test_router.py
```

- [ ] **Step 6: Commit**

```bash
git add src/router.py tests/test_router.py
git commit -m "feat: add router with RAG/agent heuristic"
```

---

## Task 3: cache.py

**Files:**
- Create: `fase5-production/src/cache.py`
- Create: `fase5-production/tests/test_cache.py`

- [ ] **Step 1: Escrever os testes**

```python
# fase5-production/tests/test_cache.py
import time
from unittest.mock import MagicMock, patch

from src.cache import SemanticCache


def _make_cache(threshold: float = 0.92, ttl: int = 86400) -> SemanticCache:
    """Cria SemanticCache sem conectar ao ChromaDB ou Ollama."""
    cache = SemanticCache.__new__(SemanticCache)
    cache._threshold = threshold
    cache._ttl = ttl
    cache._collection = MagicMock()
    return cache


def test_get_miss_no_results() -> None:
    cache = _make_cache()
    cache._collection.query.return_value = {
        "distances": [[]],
        "documents": [[]],
        "metadatas": [[]],
    }
    with patch.object(cache, "_embed", return_value=[0.1] * 768):
        assert cache.get("What is RAG?") is None


def test_get_hit_above_threshold() -> None:
    cache = _make_cache()
    cache._collection.query.return_value = {
        "distances": [[0.04]],  # similarity = 0.96 > 0.92
        "documents": [["cached answer"]],
        "metadatas": [[{"question": "q", "cached_at": time.time()}]],
    }
    with patch.object(cache, "_embed", return_value=[0.1] * 768):
        assert cache.get("What is RAG?") == "cached answer"


def test_get_miss_below_threshold() -> None:
    cache = _make_cache()
    cache._collection.query.return_value = {
        "distances": [[0.12]],  # similarity = 0.88 < 0.92
        "documents": [["cached answer"]],
        "metadatas": [[{"question": "q", "cached_at": time.time()}]],
    }
    with patch.object(cache, "_embed", return_value=[0.1] * 768):
        assert cache.get("What is RAG?") is None


def test_get_miss_expired() -> None:
    cache = _make_cache(ttl=3600)
    expired = time.time() - 7200  # 2h atrás — TTL é 1h
    cache._collection.query.return_value = {
        "distances": [[0.04]],
        "documents": [["old answer"]],
        "metadatas": [[{"question": "q", "cached_at": expired}]],
    }
    with patch.object(cache, "_embed", return_value=[0.1] * 768):
        assert cache.get("What is RAG?") is None


def test_set_stores_answer_in_document() -> None:
    cache = _make_cache()
    with patch.object(cache, "_embed", return_value=[0.1] * 768):
        cache.set("question", "my answer")
    call_kwargs = cache._collection.upsert.call_args.kwargs
    assert call_kwargs["documents"] == ["my answer"]
    assert "cached_at" in call_kwargs["metadatas"][0]


def test_set_silently_ignores_errors() -> None:
    cache = _make_cache()
    cache._collection.upsert.side_effect = Exception("chroma error")
    with patch.object(cache, "_embed", return_value=[0.1] * 768):
        cache.set("question", "answer")  # não deve levantar exceção


def test_get_returns_none_on_exception() -> None:
    cache = _make_cache()
    cache._collection.query.side_effect = Exception("chroma error")
    with patch.object(cache, "_embed", return_value=[0.1] * 768):
        assert cache.get("question") is None
```

- [ ] **Step 2: Rodar para ver FAIL**

```bash
uv run pytest tests/test_cache.py -v
```

Expected: `ImportError: cannot import name 'SemanticCache'`

- [ ] **Step 3: Implementar cache.py**

```python
# fase5-production/src/cache.py
"""Cache semântico usando ChromaDB com similaridade cosine.

Armazena respostas indexadas pelo embedding da query.
Retorna hit se similaridade cosine > CACHE_SIMILARITY_THRESHOLD e não expirada.
TTL de 24h verificado via metadata cached_at.
"""

import logging
import time
from pathlib import Path

import chromadb
import ollama

logger = logging.getLogger(__name__)

CACHE_SIMILARITY_THRESHOLD: float = 0.92
CACHE_TTL_SECONDS: int = 24 * 3600
EMBED_MODEL: str = "nomic-embed-text"
CACHE_COLLECTION: str = "query_cache"
CHROMA_PATH: Path = Path(__file__).resolve().parent.parent / "data" / "chroma_cache"


class SemanticCache:
    """Cache semântico: armazena respostas indexadas por embedding da query."""

    def __init__(
        self,
        chroma_path: Path = CHROMA_PATH,
        threshold: float = CACHE_SIMILARITY_THRESHOLD,
        ttl: int = CACHE_TTL_SECONDS,
    ) -> None:
        self._threshold = threshold
        self._ttl = ttl
        client = chromadb.PersistentClient(path=str(chroma_path))
        self._collection = client.get_or_create_collection(
            name=CACHE_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    def _embed(self, text: str) -> list[float]:
        resp = ollama.embeddings(model=EMBED_MODEL, prompt=text)
        return resp["embedding"]

    def get(self, question: str) -> str | None:
        """Retorna resposta cached se similaridade > threshold e não expirada.

        Args:
            question: pergunta do usuário.

        Returns:
            Resposta cached como string, ou None se miss/expirado.
        """
        try:
            embedding = self._embed(question)
            results = self._collection.query(
                query_embeddings=[embedding],
                n_results=1,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            return None

        if not results["distances"] or not results["distances"][0]:
            return None

        distance = results["distances"][0][0]
        similarity = 1.0 - distance

        if similarity < self._threshold:
            return None

        metadata = results["metadatas"][0][0]
        cached_at = float(metadata.get("cached_at", 0.0))
        if time.time() - cached_at > self._ttl:
            return None

        logger.info("cache_hit similarity=%.3f", similarity)
        return results["documents"][0][0]

    def set(self, question: str, answer: str) -> None:
        """Persiste uma resposta no cache.

        Args:
            question: pergunta original.
            answer: resposta a armazenar.
        """
        try:
            embedding = self._embed(question)
            doc_id = f"q_{abs(hash(question)) % (10**9)}"
            self._collection.upsert(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[answer],
                metadatas=[{"question": question, "cached_at": time.time()}],
            )
        except Exception as exc:
            logger.warning("cache_set_failed error=%s", exc)
```

- [ ] **Step 4: Rodar para ver PASS**

```bash
uv run pytest tests/test_cache.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Lint**

```bash
uv run ruff check src/cache.py tests/test_cache.py && uv run ruff format src/cache.py tests/test_cache.py
```

- [ ] **Step 6: Commit**

```bash
git add src/cache.py tests/test_cache.py
git commit -m "feat: add semantic cache with ChromaDB"
```

---

## Task 4: rag_adapter.py

**Files:**
- Create: `fase5-production/src/rag_adapter.py`
- Create: `fase5-production/tests/test_rag_adapter.py`

- [ ] **Step 1: Escrever os testes**

```python
# fase5-production/tests/test_rag_adapter.py
from unittest.mock import MagicMock, patch

from src.rag_adapter import RagAdapter


def _make_adapter() -> RagAdapter:
    """Cria RagAdapter sem conectar ao ChromaDB."""
    adapter = RagAdapter.__new__(RagAdapter)
    adapter._collection = MagicMock()
    return adapter


def test_query_calls_retrieve_and_generate() -> None:
    adapter = _make_adapter()
    fake_chunks = [{"text": "RAG content", "source": "paper.pdf", "page": 1}]
    with (
        patch("src.rag_adapter.retrieve", return_value=fake_chunks),
        patch("src.rag_adapter.generate", return_value="RAG answer"),
    ):
        result = adapter.query("What is RAG?")
    assert result == "RAG answer"


def test_query_passes_collection_to_retrieve() -> None:
    adapter = _make_adapter()
    with (
        patch("src.rag_adapter.retrieve", return_value=[]) as mock_retrieve,
        patch("src.rag_adapter.generate", return_value="answer"),
    ):
        adapter.query("question")
    mock_retrieve.assert_called_once_with("question", collection=adapter._collection)


def test_stream_yields_tokens() -> None:
    adapter = _make_adapter()
    fake_chunks = [{"text": "content", "source": "p.pdf", "page": 1}]
    fake_stream = [
        MagicMock(message=MagicMock(content="token1")),
        MagicMock(message=MagicMock(content="token2")),
        MagicMock(message=MagicMock(content="")),  # token vazio — deve ser ignorado
    ]
    with (
        patch("src.rag_adapter.retrieve", return_value=fake_chunks),
        patch("src.rag_adapter.format_context", return_value="formatted context"),
        patch("src.rag_adapter.build_prompt", return_value=[{"role": "user", "content": "q"}]),
        patch("src.rag_adapter.ollama.chat", return_value=iter(fake_stream)),
    ):
        tokens = list(adapter.stream("question"))
    assert tokens == ["token1", "token2"]


def test_stream_passes_collection_to_retrieve() -> None:
    adapter = _make_adapter()
    with (
        patch("src.rag_adapter.retrieve", return_value=[]) as mock_retrieve,
        patch("src.rag_adapter.format_context", return_value=""),
        patch("src.rag_adapter.build_prompt", return_value=[]),
        patch("src.rag_adapter.ollama.chat", return_value=iter([])),
    ):
        list(adapter.stream("question"))
    mock_retrieve.assert_called_once_with("question", collection=adapter._collection)
```

- [ ] **Step 2: Rodar para ver FAIL**

```bash
uv run pytest tests/test_rag_adapter.py -v
```

Expected: `ImportError: cannot import name 'RagAdapter'`

- [ ] **Step 3: Implementar rag_adapter.py**

```python
# fase5-production/src/rag_adapter.py
"""Adapter para o pipeline RAG da Fase 1.

Encapsula retrieve + generate da Fase 1 com o ChromaDB no caminho absoluto correto.
Não usa pipeline.query() para evitar problemas com CHROMA_PATH relativo.
"""

import sys
from collections.abc import Iterator
from pathlib import Path

import ollama

_FASE1_ROOT = Path(__file__).resolve().parent.parent.parent / "fase1-rag"
if str(_FASE1_ROOT) not in sys.path:
    sys.path.insert(0, str(_FASE1_ROOT))

from src.generation import GENERATION_MODEL, build_prompt, generate  # noqa: E402
from src.retrieval import format_context, retrieve  # noqa: E402

_CHROMA_PATH = _FASE1_ROOT / "chroma_db"
_PAPERS_COLLECTION = "papers"


class RagAdapter:
    """Encapsula o pipeline RAG da Fase 1, usando caminho absoluto para o ChromaDB."""

    def __init__(self) -> None:
        import chromadb

        client = chromadb.PersistentClient(path=str(_CHROMA_PATH))
        self._collection = client.get_or_create_collection(name=_PAPERS_COLLECTION)

    def query(self, question: str) -> str:
        """Executa retrieval + generation, retorna resposta completa.

        Args:
            question: pergunta do usuário.

        Returns:
            Resposta gerada pelo LLM.
        """
        chunks = retrieve(question, collection=self._collection)
        return generate(question, chunks)

    def stream(self, question: str) -> Iterator[str]:
        """Executa retrieval e gera tokens via Ollama streaming.

        Args:
            question: pergunta do usuário.

        Yields:
            Tokens de texto gerados pelo LLM.
        """
        chunks = retrieve(question, collection=self._collection)
        context = format_context(chunks)
        messages = build_prompt(context, question)

        stream = ollama.chat(model=GENERATION_MODEL, messages=messages, stream=True)
        for chunk in stream:
            token = chunk.message.content
            if token:
                yield token
```

- [ ] **Step 4: Rodar para ver PASS**

```bash
uv run pytest tests/test_rag_adapter.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Lint**

```bash
uv run ruff check src/rag_adapter.py tests/test_rag_adapter.py && uv run ruff format src/rag_adapter.py tests/test_rag_adapter.py
```

- [ ] **Step 6: Commit**

```bash
git add src/rag_adapter.py tests/test_rag_adapter.py
git commit -m "feat: add RagAdapter encapsulating fase1 pipeline"
```

---

## Task 5: agent_adapter.py

**Files:**
- Create: `fase5-production/src/agent_adapter.py`
- Create: `fase5-production/tests/test_agent_adapter.py`

- [ ] **Step 1: Escrever os testes**

```python
# fase5-production/tests/test_agent_adapter.py
from unittest.mock import MagicMock

from src.agent_adapter import AgentAdapter


def _make_adapter() -> AgentAdapter:
    """Cria AgentAdapter sem instanciar ReActAgent."""
    adapter = AgentAdapter.__new__(AgentAdapter)
    adapter._agent = MagicMock()
    return adapter


def test_run_returns_answer_string() -> None:
    adapter = _make_adapter()
    adapter._agent.run.return_value = MagicMock(answer="agent answer")
    result = adapter.run("Compare RAG and fine-tuning")
    assert result == "agent answer"


def test_run_passes_question_to_agent() -> None:
    adapter = _make_adapter()
    adapter._agent.run.return_value = MagicMock(answer="answer")
    adapter.run("the question")
    adapter._agent.run.assert_called_once_with("the question")


def test_run_returns_string_type() -> None:
    adapter = _make_adapter()
    adapter._agent.run.return_value = MagicMock(answer="response text")
    result = adapter.run("question")
    assert isinstance(result, str)
```

- [ ] **Step 2: Rodar para ver FAIL**

```bash
uv run pytest tests/test_agent_adapter.py -v
```

Expected: `ImportError: cannot import name 'AgentAdapter'`

- [ ] **Step 3: Implementar agent_adapter.py**

```python
# fase5-production/src/agent_adapter.py
"""Adapter para o Agente ReAct da Fase 3.

Instancia ReActAgent com as ferramentas padrão (search_papers, calculate, summarize)
e encapsula a chamada a agent.run() retornando apenas a string de resposta.
"""

import sys
from pathlib import Path

_FASE3_ROOT = Path(__file__).resolve().parent.parent.parent / "fase3-agents"
if str(_FASE3_ROOT) not in sys.path:
    sys.path.insert(0, str(_FASE3_ROOT))

from src.agent import ReActAgent  # noqa: E402
from src.tools import calculate, search_papers, summarize  # noqa: E402

_TOOLS = {
    "search_papers": search_papers,
    "calculate": calculate,
    "summarize": summarize,
}


class AgentAdapter:
    """Encapsula o Agente ReAct da Fase 3."""

    def __init__(self, model: str = "llama3.2", max_steps: int = 10) -> None:
        self._agent = ReActAgent(tools=_TOOLS, model=model, max_steps=max_steps)

    def run(self, question: str) -> str:
        """Executa o agente ReAct e retorna a resposta final.

        Args:
            question: pergunta complexa do usuário.

        Returns:
            Resposta final como string.
        """
        result = self._agent.run(question)
        return result.answer
```

- [ ] **Step 4: Rodar para ver PASS**

```bash
uv run pytest tests/test_agent_adapter.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Lint**

```bash
uv run ruff check src/agent_adapter.py tests/test_agent_adapter.py && uv run ruff format src/agent_adapter.py tests/test_agent_adapter.py
```

- [ ] **Step 6: Commit**

```bash
git add src/agent_adapter.py tests/test_agent_adapter.py
git commit -m "feat: add AgentAdapter encapsulating fase3 ReActAgent"
```

---

## Task 6: streaming.py

**Files:**
- Create: `fase5-production/src/streaming.py`
- Create: `fase5-production/tests/test_streaming.py`

- [ ] **Step 1: Escrever os testes**

```python
# fase5-production/tests/test_streaming.py
from src.streaming import SSE_DONE, string_to_sse, to_sse


def test_to_sse_formats_each_token() -> None:
    events = list(to_sse(iter(["hello", " world"])))
    assert events[0] == "data: hello\n\n"
    assert events[1] == "data:  world\n\n"


def test_to_sse_ends_with_done() -> None:
    events = list(to_sse(iter(["token"])))
    assert events[-1] == SSE_DONE


def test_to_sse_empty_iterator() -> None:
    events = list(to_sse(iter([])))
    assert events == [SSE_DONE]


def test_string_to_sse_produces_two_events() -> None:
    events = list(string_to_sse("full answer here"))
    assert len(events) == 2
    assert events[0] == "data: full answer here\n\n"
    assert events[1] == SSE_DONE


def test_sse_done_constant_format() -> None:
    assert SSE_DONE == "data: [DONE]\n\n"
```

- [ ] **Step 2: Rodar para ver FAIL**

```bash
uv run pytest tests/test_streaming.py -v
```

Expected: `ImportError: cannot import name 'SSE_DONE'`

- [ ] **Step 3: Implementar streaming.py**

```python
# fase5-production/src/streaming.py
"""Formatação de respostas como Server-Sent Events (SSE).

Formato SSE padrão:
  data: <conteúdo>\n\n

Evento de encerramento:
  data: [DONE]\n\n
"""

from collections.abc import Iterator

SSE_DONE = "data: [DONE]\n\n"


def to_sse(tokens: Iterator[str]) -> Iterator[str]:
    """Converte um iterador de tokens em eventos SSE.

    Args:
        tokens: iterador de strings (tokens do LLM).

    Yields:
        Strings no formato SSE, terminando com SSE_DONE.
    """
    for token in tokens:
        yield f"data: {token}\n\n"
    yield SSE_DONE


def string_to_sse(text: str) -> Iterator[str]:
    """Converte uma string completa em um evento SSE único + DONE.

    Usado quando a resposta já está pronta (cache hit, agent).

    Args:
        text: string completa a enviar.

    Yields:
        Evento com o texto completo, seguido de SSE_DONE.
    """
    yield f"data: {text}\n\n"
    yield SSE_DONE
```

- [ ] **Step 4: Rodar para ver PASS**

```bash
uv run pytest tests/test_streaming.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Lint**

```bash
uv run ruff check src/streaming.py tests/test_streaming.py && uv run ruff format src/streaming.py tests/test_streaming.py
```

- [ ] **Step 6: Commit**

```bash
git add src/streaming.py tests/test_streaming.py
git commit -m "feat: add SSE streaming helpers"
```

---

## Task 7: middleware.py

**Files:**
- Create: `fase5-production/src/middleware.py`
- Create: `fase5-production/tests/test_middleware.py`

- [ ] **Step 1: Escrever os testes**

```python
# fase5-production/tests/test_middleware.py
import json
import logging

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from src.middleware import LoggingMiddleware


def _make_app() -> FastAPI:
    """App mínima para testar o middleware."""
    app = FastAPI()
    app.add_middleware(LoggingMiddleware)

    @app.get("/ping")
    async def ping():
        return {"pong": True}

    @app.post("/with-state")
    async def with_state(request: Request):
        request.state.solver = "rag"
        request.state.cache_hit = False
        request.state.question_hash = "abc12345"
        return {"ok": True}

    return app


def test_request_returns_200() -> None:
    client = TestClient(_make_app())
    assert client.get("/ping").status_code == 200


def test_trace_id_is_logged(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="src.middleware"):
        TestClient(_make_app()).get("/ping")
    entries = [r for r in caplog.records if "trace_id" in r.message]
    assert len(entries) == 1
    entry = json.loads(entries[0].message)
    assert len(entry["trace_id"]) == 8


def test_log_has_required_fields(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="src.middleware"):
        TestClient(_make_app()).get("/ping")
    entries = [r for r in caplog.records if "trace_id" in r.message]
    entry = json.loads(entries[0].message)
    for field in ("trace_id", "method", "path", "status_code", "latency_ms"):
        assert field in entry, f"campo '{field}' ausente no log"


def test_optional_fields_logged_when_set(caplog) -> None:
    app = _make_app()
    with caplog.at_level(logging.INFO, logger="src.middleware"):
        TestClient(app).post("/with-state")
    entries = [r for r in caplog.records if "trace_id" in r.message]
    entry = json.loads(entries[0].message)
    assert entry["solver"] == "rag"
    assert entry["cache_hit"] is False
    assert entry["question_hash"] == "abc12345"


def test_optional_fields_absent_for_non_query(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="src.middleware"):
        TestClient(_make_app()).get("/ping")
    entries = [r for r in caplog.records if "trace_id" in r.message]
    entry = json.loads(entries[0].message)
    assert "question_hash" not in entry
    assert "solver" not in entry
```

- [ ] **Step 2: Rodar para ver FAIL**

```bash
uv run pytest tests/test_middleware.py -v
```

Expected: `ImportError: cannot import name 'LoggingMiddleware'`

- [ ] **Step 3: Implementar middleware.py**

```python
# fase5-production/src/middleware.py
"""Middleware de logging estruturado JSON com trace_id por request.

Injeta um trace_id único (8 chars hex) em cada request via request.state.
Loga ao final de cada request: trace_id, method, path, status_code, latency_ms.
Campos opcionais (question_hash, solver, cache_hit) são logados se setados pelo handler.
"""

import json
import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Injeta trace_id e loga métricas de cada request em formato JSON."""

    async def dispatch(self, request: Request, call_next) -> Response:
        trace_id = uuid.uuid4().hex[:8]
        request.state.trace_id = trace_id
        start = time.perf_counter()

        response = await call_next(request)

        latency_ms = int((time.perf_counter() - start) * 1000)

        entry: dict = {
            "trace_id": trace_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "latency_ms": latency_ms,
        }

        # Campos opcionais setados pelo handler de /query
        for field in ("question_hash", "solver", "cache_hit"):
            value = getattr(request.state, field, None)
            if value is not None:
                entry[field] = value

        logger.info(json.dumps(entry))
        return response
```

- [ ] **Step 4: Rodar para ver PASS**

```bash
uv run pytest tests/test_middleware.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Lint**

```bash
uv run ruff check src/middleware.py tests/test_middleware.py && uv run ruff format src/middleware.py tests/test_middleware.py
```

- [ ] **Step 6: Commit**

```bash
git add src/middleware.py tests/test_middleware.py
git commit -m "feat: add structured JSON logging middleware with trace_id"
```

---

## Task 8: monitor.py

**Files:**
- Create: `fase5-production/src/monitor.py`
- Create: `fase5-production/tests/test_monitor.py`

- [ ] **Step 1: Escrever os testes**

```python
# fase5-production/tests/test_monitor.py
import json
import logging
from unittest.mock import MagicMock, patch

import pytest
from fastapi import BackgroundTasks

from src.monitor import EVAL_SAMPLE_RATE, _run_eval, maybe_schedule_eval


def test_eval_scheduled_when_random_below_rate() -> None:
    background = MagicMock(spec=BackgroundTasks)
    with patch("src.monitor.random.random", return_value=0.05):  # 0.05 < 0.1
        maybe_schedule_eval("q", "a", background, "abc123")
    background.add_task.assert_called_once()


def test_eval_not_scheduled_when_random_above_rate() -> None:
    background = MagicMock(spec=BackgroundTasks)
    with patch("src.monitor.random.random", return_value=0.99):  # 0.99 > 0.1
        maybe_schedule_eval("q", "a", background, "abc123")
    background.add_task.assert_not_called()


def test_eval_sample_rate_value() -> None:
    assert EVAL_SAMPLE_RATE == 0.1


def test_run_eval_logs_relevance(caplog) -> None:
    with (
        patch("src.monitor._answer_relevance", return_value=0.85),
        patch("src.monitor._FASE2_AVAILABLE", True),
        caplog.at_level(logging.INFO, logger="src.monitor"),
    ):
        _run_eval("What is RAG?", "RAG is a retrieval technique", "abc123")

    entries = [r for r in caplog.records if "answer_relevance" in r.message]
    assert len(entries) == 1
    entry = json.loads(entries[0].message)
    assert entry["trace_id"] == "abc123"
    assert entry["answer_relevance"] == 0.85


def test_run_eval_skips_when_fase2_unavailable(caplog) -> None:
    with (
        patch("src.monitor._FASE2_AVAILABLE", False),
        caplog.at_level(logging.WARNING, logger="src.monitor"),
    ):
        _run_eval("q", "a", "abc123")

    assert any("eval_skipped" in r.message for r in caplog.records)


def test_run_eval_handles_exception(caplog) -> None:
    with (
        patch("src.monitor._answer_relevance", side_effect=Exception("ollama down")),
        patch("src.monitor._FASE2_AVAILABLE", True),
        caplog.at_level(logging.WARNING, logger="src.monitor"),
    ):
        _run_eval("q", "a", "abc123")  # não deve levantar exceção

    assert any("eval_failed" in r.message for r in caplog.records)
```

- [ ] **Step 2: Rodar para ver FAIL**

```bash
uv run pytest tests/test_monitor.py -v
```

Expected: `ImportError: cannot import name 'EVAL_SAMPLE_RATE'`

- [ ] **Step 3: Implementar monitor.py**

```python
# fase5-production/src/monitor.py
"""Monitor de qualidade em produção via sampling e BackgroundTask.

Amostra EVAL_SAMPLE_RATE (10%) dos requests e executa answer_relevance
da Fase 2 em background — sem bloquear a resposta ao usuário.
"""

import json
import logging
import random
import sys
from pathlib import Path

from fastapi import BackgroundTasks

logger = logging.getLogger(__name__)

EVAL_SAMPLE_RATE: float = 0.1

_FASE2_ROOT = Path(__file__).resolve().parent.parent.parent / "fase2-evals"
if str(_FASE2_ROOT) not in sys.path:
    sys.path.insert(0, str(_FASE2_ROOT))

try:
    from src.metrics_generation import answer_relevance as _answer_relevance  # noqa: E402

    _FASE2_AVAILABLE = True
except ImportError:
    _answer_relevance = None  # type: ignore[assignment]
    _FASE2_AVAILABLE = False


def maybe_schedule_eval(
    question: str,
    answer: str,
    background_tasks: BackgroundTasks,
    trace_id: str,
) -> None:
    """Enfileira eval em background com probabilidade EVAL_SAMPLE_RATE.

    Args:
        question: pergunta original do usuário.
        answer: resposta gerada pelo pipeline.
        background_tasks: instância de BackgroundTasks do FastAPI.
        trace_id: identificador do request para correlação nos logs.
    """
    if random.random() < EVAL_SAMPLE_RATE:
        background_tasks.add_task(_run_eval, question, answer, trace_id)


def _run_eval(question: str, answer: str, trace_id: str) -> None:
    """Executa answer_relevance da Fase 2 e loga o resultado.

    Args:
        question: pergunta original.
        answer: resposta gerada.
        trace_id: identificador do request para correlação.
    """
    if not _FASE2_AVAILABLE or _answer_relevance is None:
        logger.warning("eval_skipped trace_id=%s reason=fase2_not_available", trace_id)
        return

    try:
        relevance = _answer_relevance(answer=answer, question=question)
        logger.info(
            json.dumps({
                "trace_id": trace_id,
                "eval": True,
                "answer_relevance": round(relevance, 3),
            })
        )
    except Exception as exc:
        logger.warning("eval_failed trace_id=%s error=%s", trace_id, exc)
```

- [ ] **Step 4: Rodar para ver PASS**

```bash
uv run pytest tests/test_monitor.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Lint**

```bash
uv run ruff check src/monitor.py tests/test_monitor.py && uv run ruff format src/monitor.py tests/test_monitor.py
```

- [ ] **Step 6: Commit**

```bash
git add src/monitor.py tests/test_monitor.py
git commit -m "feat: add production eval monitor with background sampling"
```

---

## Task 9: api.py

**Files:**
- Create: `fase5-production/src/api.py`
- Create: `fase5-production/tests/test_api.py`

- [ ] **Step 1: Escrever os testes**

```python
# fase5-production/tests/test_api.py
from unittest.mock import MagicMock, patch

import pytest

import src.api as api_module


@pytest.fixture()
def client():
    """TestClient com todos os singletons mockados."""
    from fastapi.testclient import TestClient

    mock_cache = MagicMock()
    mock_rag = MagicMock()
    mock_agent = MagicMock()
    mock_cache.get.return_value = None
    mock_rag.query.return_value = "RAG answer"
    mock_rag.stream.return_value = iter(["tok1", "tok2"])
    mock_agent.run.return_value = "Agent answer"

    with (
        patch.object(api_module, "_cache", mock_cache),
        patch.object(api_module, "_rag", mock_rag),
        patch.object(api_module, "_agent", mock_agent),
        patch("src.api.maybe_schedule_eval"),
    ):
        yield TestClient(api_module.app, raise_server_exceptions=True)


def test_query_rag_returns_answer(client) -> None:
    with patch("src.api.route", return_value="rag"):
        response = client.post("/query", json={"question": "What is RAG?"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "RAG answer"
    assert data["solver"] == "rag"
    assert data["cached"] is False


def test_query_agent_returns_answer(client) -> None:
    with patch("src.api.route", return_value="agent"):
        response = client.post("/query", json={"question": "Compare RAG vs fine-tuning"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Agent answer"
    assert data["solver"] == "agent"


def test_query_returns_cache_hit(client) -> None:
    api_module._cache.get.return_value = "cached answer"
    response = client.post("/query", json={"question": "What is RAG?"})
    assert response.status_code == 200
    data = response.json()
    assert data["cached"] is True
    assert data["solver"] == "cache"
    assert data["answer"] == "cached answer"


def test_query_empty_question_returns_400(client) -> None:
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 400


def test_query_missing_question_returns_400(client) -> None:
    response = client.post("/query", json={})
    assert response.status_code == 400


def test_health_endpoint_structure(client) -> None:
    with (
        patch("src.api._check_ollama", return_value=True),
        patch("src.api._check_chromadb", return_value=True),
    ):
        response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "ollama" in data
    assert "chromadb" in data
    assert data["status"] == "ok"


def test_health_degraded_when_ollama_down(client) -> None:
    with (
        patch("src.api._check_ollama", return_value=False),
        patch("src.api._check_chromadb", return_value=True),
    ):
        response = client.get("/health")
    assert response.json()["status"] == "degraded"
```

- [ ] **Step 2: Rodar para ver FAIL**

```bash
uv run pytest tests/test_api.py -v
```

Expected: `ImportError: cannot import name 'app' from 'src.api'`

- [ ] **Step 3: Implementar api.py**

```python
# fase5-production/src/api.py
"""FastAPI app — endpoints POST /query e GET /health.

Fluxo do /query:
1. Middleware (LoggingMiddleware) injeta trace_id
2. Cache semântico verifica se há hit
3. Router decide RAG vs Agent
4. RAG+SSE: streaming token a token; demais: JSON buffered
5. Cache.set() persiste a resposta
6. monitor.maybe_schedule_eval() enfileira eval em background (10% dos requests)
"""

import hashlib
import logging
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from src.agent_adapter import AgentAdapter
from src.cache import SemanticCache
from src.middleware import LoggingMiddleware
from src.monitor import maybe_schedule_eval
from src.rag_adapter import RagAdapter
from src.router import route
from src.streaming import string_to_sse, to_sse

logger = logging.getLogger(__name__)

_cache: SemanticCache | None = None
_rag: RagAdapter | None = None
_agent: AgentAdapter | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicializa singletons na subida da aplicação."""
    global _cache, _rag, _agent
    _cache = SemanticCache()
    _rag = RagAdapter()
    _agent = AgentAdapter()
    yield


app = FastAPI(title="AI Engineering Journey — Fase 5", lifespan=lifespan)
app.add_middleware(LoggingMiddleware)


def _check_ollama() -> bool:
    """Verifica se o Ollama está respondendo."""
    try:
        import ollama as _ollama

        _ollama.list()
        return True
    except Exception:
        return False


def _check_chromadb() -> bool:
    """Verifica se o ChromaDB está acessível."""
    try:
        import chromadb

        chromadb.PersistentClient(path="data/chroma_health_check")
        return True
    except Exception:
        return False


@app.post("/query")
async def query_endpoint(request: Request, background_tasks: BackgroundTasks):
    """Responde uma pergunta usando RAG ou o agente ReAct.

    Body: {"question": "<pergunta>"}
    Headers: Accept: text/event-stream para streaming SSE (somente RAG).

    Returns:
        JSON: {"answer": str, "solver": str, "cached": bool}
        SSE: stream de tokens seguido de "data: [DONE]"
    """
    body = await request.json()
    question = body.get("question", "").strip()
    if not question:
        return JSONResponse({"error": "question is required"}, status_code=400)

    question_hash = hashlib.sha256(question.encode()).hexdigest()[:8]
    request.state.question_hash = question_hash
    trace_id = getattr(request.state, "trace_id", "unknown")
    wants_sse = "text/event-stream" in request.headers.get("accept", "")

    # Cache lookup
    cached = _cache.get(question)
    if cached is not None:
        request.state.solver = "cache"
        request.state.cache_hit = True
        maybe_schedule_eval(question, cached, background_tasks, trace_id)
        if wants_sse:
            return StreamingResponse(string_to_sse(cached), media_type="text/event-stream")
        return JSONResponse({"answer": cached, "solver": "cache", "cached": True})

    request.state.cache_hit = False
    solver = route(question)
    request.state.solver = solver

    # RAG com streaming real (sem cache para este caminho)
    if solver == "rag" and wants_sse:
        return StreamingResponse(
            to_sse(_rag.stream(question)),
            media_type="text/event-stream",
        )

    # Não-streaming ou agente: buferiza para cache + eval
    answer = _agent.run(question) if solver == "agent" else _rag.query(question)
    _cache.set(question, answer)
    maybe_schedule_eval(question, answer, background_tasks, trace_id)

    if wants_sse:  # agent + SSE: resposta única
        return StreamingResponse(string_to_sse(answer), media_type="text/event-stream")
    return JSONResponse({"answer": answer, "solver": solver, "cached": False})


@app.get("/health")
async def health():
    """Verifica liveness do Ollama e ChromaDB."""
    ollama_ok = _check_ollama()
    chroma_ok = _check_chromadb()
    status = "ok" if (ollama_ok and chroma_ok) else "degraded"
    return {"status": status, "ollama": ollama_ok, "chromadb": chroma_ok}
```

- [ ] **Step 4: Rodar para ver PASS**

```bash
uv run pytest tests/test_api.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Rodar todos os testes unitários**

```bash
uv run pytest tests/ -v --ignore=tests/test_integration.py
```

Expected: todos os testes passando (36+ unitários).

- [ ] **Step 6: Lint**

```bash
uv run ruff check src/ tests/ && uv run ruff format src/ tests/
```

- [ ] **Step 7: Commit**

```bash
git add src/api.py tests/test_api.py
git commit -m "feat: add FastAPI app with /query and /health endpoints"
```

---

## Task 10: test_integration.py

**Files:**
- Create: `fase5-production/tests/test_integration.py`

Requer Ollama rodando (`ollama serve`) e fase1-rag indexada (ChromaDB populado).

- [ ] **Step 1: Escrever os testes de integração**

```python
# fase5-production/tests/test_integration.py
"""Testes de integração — requerem Ollama rodando e ChromaDB da Fase 1 populado.

Executar com:
    uv run pytest tests/test_integration.py -v -m integration
"""

import pytest
from fastapi.testclient import TestClient

from src.api import app

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_check_all_services_ok(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["ollama"] is True, "Ollama não está rodando — execute: ollama serve"
    assert data["chromadb"] is True


def test_query_rag_returns_non_empty_answer(client) -> None:
    response = client.post("/query", json={"question": "What is RAG?"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["answer"]) > 20
    assert data["solver"] in ("rag", "cache")
    assert isinstance(data["cached"], bool)


def test_query_agent_routes_for_comparison(client) -> None:
    response = client.post(
        "/query",
        json={"question": "Compare RAG and fine-tuning approaches"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["solver"] in ("agent", "cache")


def test_second_query_may_hit_cache(client) -> None:
    question = "What is attention mechanism in transformers?"
    # Primeira chamada — popula cache
    r1 = client.post("/query", json={"question": question})
    assert r1.status_code == 200

    # Segunda chamada — pode ou não vir do cache dependendo do threshold
    r2 = client.post("/query", json={"question": question})
    assert r2.status_code == 200
    # Se vier do cache, cached deve ser True
    if r2.json()["solver"] == "cache":
        assert r2.json()["cached"] is True
```

- [ ] **Step 2: Verificar que testes unitários ainda passam sem a flag de integração**

```bash
uv run pytest tests/ -v -m "not integration"
```

Expected: todos os unitários passando, os de integração pulados.

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration tests for /query and /health"
```

---

## Task 11: Docker

**Files:**
- Create: `fase5-production/Dockerfile`
- Create: `fase5-production/docker-compose.yml`
- Create: `fase5-production/.dockerignore`

- [ ] **Step 1: Criar .dockerignore**

```
# fase5-production/.dockerignore
**/.venv
**/__pycache__
**/*.pyc
**/.pytest_cache
**/chroma_db
**/data/chroma_cache
**/.git
```

- [ ] **Step 2: Criar Dockerfile**

O build context é a raiz de `ai-engineering-journey/` (contém todas as fases).

```dockerfile
# fase5-production/Dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

RUN pip install uv

WORKDIR /app

# Copiar fases necessárias (contexto = raiz de ai-engineering-journey/)
COPY fase1-rag/ ./fase1-rag/
COPY fase2-evals/ ./fase2-evals/
COPY fase3-agents/ ./fase3-agents/
COPY fase5-production/ ./fase5-production/

# Instalar dependências da fase5
WORKDIR /app/fase5-production
RUN uv sync --no-dev

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 3: Criar docker-compose.yml**

```yaml
# fase5-production/docker-compose.yml
services:
  app:
    build:
      context: ..
      dockerfile: fase5-production/Dockerfile
    ports:
      - "8000:8000"
    volumes:
      # Monta o ChromaDB da Fase 1 como read-only
      - ../fase1-rag/chroma_db:/app/fase1-rag/chroma_db:ro
      # Dados do cache semântico persistem entre reinicializações
      - fase5_cache:/app/fase5-production/data/chroma_cache
    depends_on:
      ollama:
        condition: service_healthy
    environment:
      - PYTHONUNBUFFERED=1
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 20s

  ollama:
    image: ollama/ollama:latest
    volumes:
      - ollama_data:/root/.ollama
    ports:
      - "11434:11434"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/tags"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 10s

volumes:
  ollama_data:
  fase5_cache:
```

- [ ] **Step 4: Verificar que o docker-compose.yml está correto (dry-run)**

```bash
cd fase5-production && docker compose config
```

Expected: configuração YAML expandida sem erros.

- [ ] **Step 5: Commit**

```bash
git add Dockerfile docker-compose.yml .dockerignore
git commit -m "feat: add Dockerfile and docker-compose for fase5"
```

---

## Task 12: Lint final, Obsidian e git

- [ ] **Step 1: Rodar lint em toda a fase5**

```bash
cd fase5-production && uv run ruff check . && uv run ruff format .
```

Expected: nenhum erro de lint.

- [ ] **Step 2: Rodar todos os testes unitários**

```bash
uv run pytest tests/ -m "not integration" -v --tb=short
```

Expected: ≥ 40 testes passando.

- [ ] **Step 3: Atualizar Obsidian — criar arquivo da Fase 5**

Criar `/home/lbr/obsidian-vault/AI Engineering Journey/Fase 5 - Produção/02 - O que Aprendi.md`
com pelo menos 200 linhas cobrindo (ver CLAUDE.md global para formato obrigatório):
- FastAPI: por que existe, como funciona o ASGI, lifespan vs startup/shutdown
- Adapter Pattern: problema que resolve, como funciona, quando usar
- Server-Sent Events: o que é, como difere de WebSockets, formato SSE
- Cache Semântico: diferença de cache por hash vs por similaridade, threshold
- Logging estruturado: por que JSON, trace_id, observabilidade em produção
- BackgroundTasks: fire-and-forget no FastAPI, quando usar vs celery/redis
- Docker multi-stage: build context, volumes, healthchecks

- [ ] **Step 4: Rodar sync do Obsidian**

```bash
~/sync-obsidian.sh
```

- [ ] **Step 5: Atualizar NEXT_STEPS.md**

Em `docs/NEXT_STEPS.md`, marcar Fase 5 como `✅ Completo` e remover a seção `⚙️` da Fase 4.

- [ ] **Step 6: Commit final**

```bash
cd /home/lbr/projetos/ai-engineering-journey
git add fase5-production/ docs/NEXT_STEPS.md
git commit -m "feat: fase5-production completo — FastAPI + streaming + cache + Docker"
```

---

## Checklist de métricas de sucesso

Antes de marcar como concluído, verificar:

- [ ] `uv run pytest tests/ -m "not integration"` — ≥ 40 testes, 0 falhas
- [ ] `uv run ruff check . && uv run ruff format .` — sem erros de lint
- [ ] `curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"question":"What is RAG?"}'` — resposta JSON com `answer`, `solver`, `cached`
- [ ] `curl -X POST http://localhost:8000/query -H "Accept: text/event-stream" -H "Content-Type: application/json" -d '{"question":"What is RAG?"}'` — stream SSE com tokens + `[DONE]`
- [ ] `curl http://localhost:8000/health` — `{"status": "ok", "ollama": true, "chromadb": true}`
- [ ] `docker compose config` — sem erros de configuração
