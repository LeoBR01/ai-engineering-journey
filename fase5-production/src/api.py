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
            return StreamingResponse(
                string_to_sse(cached), media_type="text/event-stream"
            )
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
