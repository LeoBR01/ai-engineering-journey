"""Módulo de geração de resposta via LLM.

Responsabilidade: montar o prompt com o contexto recuperado e a pergunta
do usuário, enviar ao Ollama e retornar a resposta gerada. Este módulo
existe para isolar a lógica de prompt engineering e a integração com o LLM.

O prompt segue o padrão RAG: instrução de sistema + contexto (chunks
recuperados) + pergunta do usuário. O modelo é instruído a responder
apenas com base no contexto fornecido, evitando alucinações.
"""

import ollama

from src.retrieval import format_context

# Configurações
GENERATION_MODEL = "llama3.2"

SYSTEM_PROMPT = """Você é um assistente especializado em responder perguntas \
com base em documentos fornecidos. Responda APENAS usando as informações \
do contexto abaixo. Se a resposta não estiver no contexto, diga claramente \
que não encontrou a informação nos documentos.

Contexto:
{context}
"""


def build_prompt(context: str, question: str) -> list[dict[str, str]]:
    """Monta a lista de mensagens no formato esperado pelo Ollama.

    Args:
        context: texto com os chunks recuperados, formatado por
            retrieval.format_context().
        question: pergunta do usuário.

    Returns:
        Lista de dicts com 'role' e 'content' (formato chat do Ollama).
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
        {"role": "user", "content": question},
    ]


def generate(
    query: str,
    chunks: list[dict[str, str | int | float]],
    model: str = GENERATION_MODEL,
) -> str:
    """Gera uma resposta usando o Ollama com base nos chunks e na pergunta.

    Formata os chunks em contexto, monta o prompt e chama o LLM via Ollama.
    O modelo é instruído a responder apenas com base no contexto fornecido.

    Args:
        query: pergunta do usuário.
        chunks: chunks recuperados pelo retrieval.retrieve().
        model: nome do modelo Ollama para geração.

    Returns:
        Resposta gerada pelo LLM como string.
    """
    context = format_context(chunks)
    messages = build_prompt(context, query)
    response = ollama.chat(model=model, messages=messages)
    return response.message.content
