"""
Métricas de avaliação da geração via LLM-as-judge.

Usa um LLM (llama3.2 via Ollama) como árbitro para avaliar duas dimensões:
- Faithfulness: a resposta gerada está suportada pelo contexto recuperado?
- Answer Relevance: a resposta de fato responde à pergunta feita?

Cada métrica retorna um score de 0 a 1 baseado no julgamento do LLM.
"""

import ollama

# Modelo usado como juiz — o mesmo da Fase 1
JUDGE_MODEL = "llama3.2"

FAITHFULNESS_PROMPT = """\
Você é um avaliador rigoroso de sistemas de perguntas e respostas.

Sua tarefa: verificar se a RESPOSTA está completamente suportada
pelo CONTEXTO fornecido.
Não use conhecimento externo — julgue APENAS com base no que está no contexto.

CONTEXTO:
{context}

RESPOSTA GERADA:
{answer}

Avalie a faithfulness em uma escala de 0 a 1:
- 1.0: toda afirmação da resposta está explicitamente no contexto
- 0.5: parte da resposta está no contexto, parte não
- 0.0: a resposta contradiz ou vai além do contexto

Responda APENAS com um número decimal entre 0 e 1, sem explicação."""

RELEVANCE_PROMPT = """\
Você é um avaliador rigoroso de sistemas de perguntas e respostas.

Sua tarefa: verificar se a RESPOSTA responde adequadamente à PERGUNTA.

PERGUNTA:
{question}

RESPOSTA GERADA:
{answer}

Avalie a relevância em uma escala de 0 a 1:
- 1.0: a resposta responde completamente à pergunta
- 0.5: a resposta responde parcialmente
- 0.0: a resposta não responde à pergunta

Responda APENAS com um número decimal entre 0 e 1, sem explicação."""


def _parse_score(raw: str) -> float:
    """
    Extrai o score numérico da resposta do LLM.

    Args:
        raw: Texto bruto retornado pelo LLM.

    Returns:
        Float entre 0.0 e 1.0. Retorna 0.0 em caso de falha de parse.
    """
    try:
        score = float(raw.strip())
        return max(0.0, min(1.0, score))
    except ValueError:
        # Tenta extrair o primeiro número caso o modelo tenha adicionado texto extra
        import re

        match = re.search(r"\d+(\.\d+)?", raw)
        if match:
            return max(0.0, min(1.0, float(match.group())))
        return 0.0


def faithfulness(answer: str, context: str, model: str = JUDGE_MODEL) -> float:
    """
    Avalia se a resposta gerada está suportada pelo contexto recuperado.

    Args:
        answer: Resposta gerada pelo pipeline RAG.
        context: Contexto recuperado pelo retrieval (chunks concatenados).
        model: Nome do modelo Ollama a usar como juiz.

    Returns:
        Score de 0.0 a 1.0.
    """
    prompt = FAITHFULNESS_PROMPT.format(context=context, answer=answer)
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response["message"]["content"]
    return _parse_score(raw)


def answer_relevance(answer: str, question: str, model: str = JUDGE_MODEL) -> float:
    """
    Avalia se a resposta gerada responde adequadamente à pergunta.

    Args:
        answer: Resposta gerada pelo pipeline RAG.
        question: Pergunta original do usuário.
        model: Nome do modelo Ollama a usar como juiz.

    Returns:
        Score de 0.0 a 1.0.
    """
    prompt = RELEVANCE_PROMPT.format(question=question, answer=answer)
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response["message"]["content"]
    return _parse_score(raw)
