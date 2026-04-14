"""Adapter para o Agente ReAct da Fase 3.

Instancia ReActAgent com as ferramentas padrão (search_papers, calculate, summarize)
e encapsula a chamada a agent.run() retornando apenas a string de resposta.
"""

import sys
from pathlib import Path

_FASE3_ROOT = Path(__file__).resolve().parent.parent.parent / "fase3-agents"
if str(_FASE3_ROOT) not in sys.path:
    sys.path.insert(0, str(_FASE3_ROOT))

# Mesma técnica do rag_adapter para evitar conflito de namespace com src
_saved_src_modules = {
    k: v for k, v in sys.modules.items() if k == "src" or k.startswith("src.")
}
for _k in list(_saved_src_modules.keys()):
    del sys.modules[_k]

try:
    from src.agent import ReActAgent  # noqa: E402
    from src.tools import calculate, search_papers, summarize  # noqa: E402
finally:
    sys.modules.update(_saved_src_modules)

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
