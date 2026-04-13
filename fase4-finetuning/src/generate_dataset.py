"""Geração de dataset de fine-tuning para llama3.2.

Produz dois tipos de exemplos em formato ChatML:
- RAG pairs: (context, pergunta, resposta ideal) gerados dos chunks do ChromaDB
- ReAct cycles: ciclos Thought→Action→Observation→Final Answer bem-formados

Uso:
    uv run generate-dataset --rag-output data/raw/rag_pairs.jsonl \\
                            --react-output data/raw/react_cycles.jsonl \\
                            --processed-output data/processed/train.jsonl
"""

import json
import logging
import sys
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


def preprocess_dataset(
    rag_path: str,
    react_path: str,
    output_path: str,
    rag_ratio: float = 0.6,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, int]:
    """Combina, balanceia e divide os datasets RAG e ReAct.

    Realiza shuffle determinístico e split treino/validação.

    Args:
        rag_path: caminho para rag_pairs.jsonl.
        react_path: caminho para react_cycles.jsonl.
        output_path: caminho para train.jsonl de saída.
        rag_ratio: fração do total a ser preenchida com exemplos RAG.
        val_ratio: fração a ser separada para validação.
        seed: semente para reprodutibilidade.

    Returns:
        Dict com contagens {'train': N, 'val': M}.
    """
    import random

    def _read_jsonl(path: str) -> list[dict]:
        # Usar split("\n") em vez de splitlines() para evitar que caracteres
        # Unicode U+2028/U+2029 (line separators) gerados pelo LLM com
        # ensure_ascii=False partam um objeto JSON no meio.
        return [
            json.loads(line)
            for line in Path(path).read_text(encoding="utf-8").split("\n")
            if line.strip()
        ]

    rag_examples = _read_jsonl(rag_path)
    react_examples = _read_jsonl(react_path)

    # Balancear proporções: calcular targets com base no menor lado disponível
    total = len(rag_examples) + len(react_examples)
    target_rag = int(total * rag_ratio)
    target_react = total - target_rag

    random.seed(seed)
    # Ajustar se um dos lados não tem exemplos suficientes
    if len(rag_examples) < target_rag:
        target_rag = len(rag_examples)
        target_react = min(len(react_examples), total - target_rag)
    elif len(react_examples) < target_react:
        target_react = len(react_examples)
        target_rag = min(len(rag_examples), total - target_react)

    if len(rag_examples) > target_rag:
        rag_examples = random.sample(rag_examples, target_rag)
    if len(react_examples) > target_react:
        react_examples = random.sample(react_examples, target_react)

    all_examples = rag_examples + react_examples
    random.shuffle(all_examples)

    # Split treino/validação
    val_size = int(len(all_examples) * val_ratio)
    val_examples = all_examples[:val_size]
    train_examples = all_examples[val_size:]

    # Salvar
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    val_path = out_path.parent / "val.jsonl"

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in train_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for ex in val_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    return {"train": len(train_examples), "val": len(val_examples)}


def _make_default_llm_fn(model: str = GENERATION_MODEL) -> Callable[[str], str]:
    """Cria uma função LLM usando ollama.chat."""
    import ollama

    def llm_fn(prompt: str) -> str:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.message.content

    return llm_fn


def _make_default_search_fn() -> Callable[[str], list[dict]]:
    """Cria a função de busca usando o search_papers da Fase 3."""
    fase3_src = Path(__file__).resolve().parent.parent.parent / "fase3-agents" / "src"
    if str(fase3_src) not in sys.path:
        sys.path.insert(0, str(fase3_src))

    from tools import search_papers  # noqa: PLC0415

    return search_papers


def _make_default_generate_fn(
    model: str = GENERATION_MODEL,
) -> Callable[[str, str], str]:
    """Cria a função generate para ciclos ReAct."""
    import ollama

    def generate_fn(question: str, context: str) -> str:
        system = (
            "Answer the question based only on the provided context. "
            "Be concise and factual.\n\nContext:\n" + context
        )
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": question},
            ],
        )
        return response.message.content

    return generate_fn


def generate_rag_dataset(
    output_path: str,
    chroma_path: str = CHROMA_PATH,
    collection_name: str = COLLECTION_NAME,
    max_pairs: int = 800,
    sample_every: int = 5,
    model: str = GENERATION_MODEL,
    llm_fn: Callable[[str], str] | None = None,
) -> int:
    """Gera o dataset de pares RAG a partir do ChromaDB da Fase 1.

    Args:
        output_path: caminho para salvar rag_pairs.jsonl.
        chroma_path: caminho para o ChromaDB.
        collection_name: nome da collection no ChromaDB.
        max_pairs: número máximo de pares a gerar.
        sample_every: amostrar 1 em cada N chunks (para reduzir tempo).
        model: nome do modelo Ollama.
        llm_fn: função LLM para injeção em testes (usa Ollama se None).

    Returns:
        Número de pares gerados e salvos.
    """
    import chromadb

    if llm_fn is None:
        llm_fn = _make_default_llm_fn(model)

    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_collection(name=collection_name)
    all_docs = collection.get(include=["documents"])

    documents = all_docs.get("documents", [])
    sampled = documents[::sample_every][:max_pairs]

    pairs = []
    for i, chunk in enumerate(sampled):
        logger.info("RAG pair %d/%d", i + 1, len(sampled))
        pair = generate_rag_pair(chunk, llm_fn)
        if pair:
            pairs.append(pair_to_chatml_rag(pair))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    return len(pairs)


def generate_react_dataset(
    output_path: str,
    questions: list[str] | None = None,
    search_fn: Callable[[str], list[dict]] | None = None,
    generate_fn: Callable[[str, str], str] | None = None,
    model: str = GENERATION_MODEL,
) -> int:
    """Gera o dataset de ciclos ReAct usando busca real + LLM.

    Args:
        output_path: caminho para salvar react_cycles.jsonl.
        questions: lista de perguntas seed (usa default se None).
        search_fn: função de busca (usa search_papers da Fase 3 se None).
        generate_fn: função de geração (usa Ollama se None).
        model: nome do modelo Ollama.

    Returns:
        Número de ciclos gerados e salvos.
    """
    if questions is None:
        questions = _default_react_questions()

    if search_fn is None:
        search_fn = _make_default_search_fn()

    if generate_fn is None:
        generate_fn = _make_default_generate_fn(model)

    # Importar SYSTEM_PROMPT da Fase 3
    fase3_src = Path(__file__).resolve().parent.parent.parent / "fase3-agents" / "src"
    if str(fase3_src) not in sys.path:
        sys.path.insert(0, str(fase3_src))

    from prompt import SYSTEM_PROMPT  # noqa: PLC0415

    cycles = []
    for i, question in enumerate(questions):
        logger.info("ReAct cycle %d/%d: %s", i + 1, len(questions), question[:50])
        cycle = generate_react_cycle(question, search_fn, generate_fn, SYSTEM_PROMPT)
        cycles.append(cycle)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for c in cycles:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    return len(cycles)


def _default_react_questions() -> list[str]:
    """Carrega perguntas do dataset da Fase 2 como seed para ciclos ReAct."""
    fase2_dataset = (
        Path(__file__).resolve().parent.parent.parent
        / "fase2-evals"
        / "data"
        / "eval_dataset.json"
    )
    if fase2_dataset.exists():
        entries = json.loads(fase2_dataset.read_text(encoding="utf-8"))
        return [e["question"] for e in entries]

    # Fallback se o dataset não estiver disponível
    return [
        "What are the main advantages of retrieval-augmented generation?",
        "How does the attention mechanism work in transformer models?",
        "What metrics are commonly used to evaluate language models?",
        "How does LoRA reduce the number of trainable parameters?",
        "What is the difference between RAG and fine-tuning?",
    ]


def main() -> None:
    """CLI para geração do dataset completo."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Gera dataset de fine-tuning")
    parser.add_argument("--rag-output", default="data/raw/rag_pairs.jsonl")
    parser.add_argument("--react-output", default="data/raw/react_cycles.jsonl")
    parser.add_argument("--processed-output", default="data/processed/train.jsonl")
    parser.add_argument("--max-rag-pairs", type=int, default=800)
    parser.add_argument("--model", default=GENERATION_MODEL)
    args = parser.parse_args()

    logger.info("Gerando RAG pairs...")
    n_rag = generate_rag_dataset(
        args.rag_output, max_pairs=args.max_rag_pairs, model=args.model
    )
    logger.info("RAG pairs gerados: %d", n_rag)

    logger.info("Gerando ciclos ReAct...")
    n_react = generate_react_dataset(args.react_output, model=args.model)
    logger.info("Ciclos ReAct gerados: %d", n_react)

    logger.info("Preprocessando dataset...")
    counts = preprocess_dataset(
        args.rag_output, args.react_output, args.processed_output
    )
    logger.info(
        "Dataset final: %d treino, %d validação", counts["train"], counts["val"]
    )
