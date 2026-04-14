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

# ─── Importação dos módulos da Fase 1 ─────────────────────────────────────────
# Ambos os projetos possuem um pacote chamado `src`. Para evitar que o Python
# resolva `from src.generation import ...` como o src desta aplicação, salvamos
# o pacote `src` atual, removemos temporariamente do sys.modules, importamos os
# símbolos necessários da fase1-rag e depois restauramos o estado original.
if str(_FASE1_ROOT) not in sys.path:
    sys.path.insert(0, str(_FASE1_ROOT))

_saved_src_modules = {
    k: v for k, v in sys.modules.items() if k == "src" or k.startswith("src.")
}
for _k in list(_saved_src_modules.keys()):
    del sys.modules[_k]

try:
    from src.generation import GENERATION_MODEL, build_prompt, generate  # noqa: E402
    from src.retrieval import format_context, retrieve  # noqa: E402
finally:
    # Restaura os módulos src da fase5 (sobrescreve qualquer src.* da fase1
    # que tenha sido carregado, exceto os que precisamos manter acessíveis).
    sys.modules.update(_saved_src_modules)

# ──────────────────────────────────────────────────────────────────────────────

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
