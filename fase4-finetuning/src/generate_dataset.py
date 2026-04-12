"""Geração de dataset de fine-tuning para llama3.2.

Produz dois tipos de exemplos em formato ChatML:
- RAG pairs: (context, pergunta, resposta ideal) gerados dos chunks do ChromaDB
- ReAct cycles: ciclos Thought→Action→Observation→Final Answer bem-formados

Uso:
    uv run generate-dataset --rag-output data/raw/rag_pairs.jsonl \\
                            --react-output data/raw/react_cycles.jsonl \\
                            --processed-output data/processed/train.jsonl
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Caminho para o ChromaDB da Fase 1
FASE1_ROOT = Path(__file__).resolve().parent.parent.parent / "fase1-rag"
CHROMA_PATH = str(FASE1_ROOT / "chroma_db")
COLLECTION_NAME = "papers"
EMBEDDING_MODEL = "nomic-embed-text"
GENERATION_MODEL = "llama3.2"

# Parâmetros de filtragem
MIN_ANSWER_LEN = 50
MIN_QUESTION_LEN = 10


@dataclass
class RagPair:
    """Par (context, pergunta, resposta) para fine-tuning RAG."""

    context: str
    question: str
    answer: str


def filter_rag_pair(pair: RagPair) -> bool:
    """Verifica se um par RAG tem qualidade mínima aceitável.

    Args:
        pair: par a ser avaliado.

    Returns:
        True se o par passou nos filtros de qualidade.
    """
    return len(pair.question) >= MIN_QUESTION_LEN and len(pair.answer) >= MIN_ANSWER_LEN


def generate_rag_pair(
    chunk: str,
    llm_fn: Callable[[str], str],
) -> RagPair | None:
    """Gera um par (pergunta, resposta) a partir de um chunk de texto.

    Chama llm_fn com um prompt pedindo uma pergunta e resposta baseadas
    exclusivamente no chunk. Parseia o formato QUESTION:/ANSWER:.

    Args:
        chunk: texto do chunk recuperado do ChromaDB.
        llm_fn: função que recebe um prompt e retorna texto do LLM.

    Returns:
        RagPair se o parse e filtros passaram, None caso contrário.
    """
    prompt = (
        "Based on the following text excerpt, generate:\n"
        "1. A specific question answerable ONLY from this text\n"
        "2. A concise, accurate answer using ONLY this text\n\n"
        f"Text: {chunk}\n\n"
        "Respond in this exact format:\n"
        "QUESTION: <your question>\n"
        "ANSWER: <your answer>"
    )

    response = llm_fn(prompt)

    if "QUESTION:" not in response or "ANSWER:" not in response:
        return None

    question = ""
    answer = ""
    for line in response.strip().splitlines():
        if line.startswith("QUESTION:"):
            question = line[len("QUESTION:") :].strip()
        elif line.startswith("ANSWER:"):
            answer = line[len("ANSWER:") :].strip()

    pair = RagPair(context=chunk, question=question, answer=answer)
    return pair if filter_rag_pair(pair) else None


def pair_to_chatml_rag(pair: RagPair) -> dict:
    """Converte um RagPair para o formato ChatML multi-turn.

    Args:
        pair: par RAG com context, question e answer.

    Returns:
        Dict com chave 'messages' no formato ChatML.
    """
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful research assistant. "
                    "Answer questions based only on the provided context."
                ),
            },
            {
                "role": "user",
                "content": f"Context: {pair.context}\n\nQuestion: {pair.question}",
            },
            {"role": "assistant", "content": pair.answer},
        ]
    }


def format_observation(search_results: list[dict]) -> str:
    """Formata resultados do search_papers como string de observação.

    Args:
        search_results: lista retornada por search_papers (dicts com 'content').

    Returns:
        Texto formatado para uso como Observation no ciclo ReAct.
    """
    if not search_results:
        return "Nenhum resultado encontrado."

    parts = []
    for i, result in enumerate(search_results[:3], 1):
        content = result.get("content", "")
        parts.append(f"[{i}] {content[:300]}")
    return "\n\n".join(parts)


def generate_react_cycle(
    question: str,
    search_fn: Callable[[str], list[dict]],
    generate_fn: Callable[[str, str], str],
    system_prompt: str,
) -> dict:
    """Gera um ciclo ReAct completo e bem-formado para fine-tuning.

    Em vez de capturar saídas do modelo atual (que tem formato inconsistente),
    constrói o ciclo programaticamente usando resultados reais do search_papers
    e respostas geradas pelo LLM. Isso garante exemplos com formato perfeito.

    Args:
        question: pergunta que o agente deve responder.
        search_fn: função de busca semântica (retorna list[dict] com 'content').
        generate_fn: função que gera resposta dado (question, context).
        system_prompt: prompt de sistema para o papel 'system' do ChatML.

    Returns:
        Dict no formato ChatML com ciclo ReAct completo.
    """
    # Executar busca real para obter observação autêntica
    search_results = search_fn(question)
    observation = format_observation(search_results)

    # Gerar resposta final baseada no contexto recuperado
    final_answer = generate_fn(question, observation)

    # Construir ciclo com formato perfeito
    assistant_content = (
        "Thought: I need to search the research papers to find relevant "
        f"information about this question.\n"
        f"Action: search_papers\n"
        f"Action Input: {question}\n"
        f"Observation: {observation}\n"
        "Thought: Based on the search results, I have enough information "
        "to provide an accurate answer.\n"
        f"Final Answer: {final_answer}"
    )

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def pair_to_chatml_react(cycle: dict) -> dict:
    """Passa o ciclo direto — já está em formato ChatML.

    Existe para simetria com pair_to_chatml_rag e para facilitar testes.

    Args:
        cycle: dict já no formato ChatML retornado por generate_react_cycle.

    Returns:
        O mesmo dict (identidade).
    """
    return cycle
