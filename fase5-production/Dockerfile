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
