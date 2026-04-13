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
