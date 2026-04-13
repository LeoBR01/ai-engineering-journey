"""Ferramentas disponíveis para o agente ReAct.

Cada ferramenta é uma função pura que recebe um input de texto e retorna
uma string com o resultado. O agente chama essas funções com base na decisão
do LLM durante o loop ReAct.

Ferramentas disponíveis:
- search_papers: busca semântica no ChromaDB da Fase 1
- calculate: avalia expressões matemáticas de forma segura via AST
- summarize: resume um texto longo usando o llama3.2 via Ollama
"""

import ast
import logging
import operator
from pathlib import Path

import chromadb
import ollama

logger = logging.getLogger(__name__)

# Configurações — mesmos valores da Fase 1
CHROMA_PATH = str(
    Path(__file__).resolve().parent.parent.parent / "fase1-rag" / "chroma_db"
)
COLLECTION_NAME = "papers"
EMBEDDING_MODEL = "nomic-embed-text"
GENERATION_MODEL = "llama3.2"

# Operações AST permitidas no calculate (apenas aritmética)
_SAFE_OPS: dict = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node: ast.AST) -> float:
    """Avalia recursivamente um nó AST, permitindo apenas operações numéricas.

    Args:
        node: nó da árvore sintática abstrata gerada pelo ast.parse().

    Returns:
        Valor numérico resultante da expressão.

    Raises:
        ValueError: se a expressão contiver qualquer operação não permitida.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return _SAFE_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"Operação não permitida: {ast.dump(node)}")


def search_papers(query: str, k: int = 5) -> list[dict]:
    """Busca os papers mais relevantes para uma query no ChromaDB da Fase 1.

    Gera o embedding da query com nomic-embed-text e executa busca por
    similaridade cosine na collection 'papers'. Reutiliza o índice já
    construído na Fase 1 sem recriar nada.

    Args:
        query: texto de busca em linguagem natural.
        k: número máximo de resultados a retornar.

    Returns:
        Lista de dicts com 'content' e 'metadata' (source, page, score).
        Retorna lista vazia se a collection estiver vazia.
    """
    logger.debug("search_papers: query=%r k=%d", query, k)

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)

    n_results = min(k, collection.count())
    if n_results == 0:
        return []

    response = ollama.embed(model=EMBEDDING_MODEL, input=query)
    embedding = response.embeddings[0]

    results = collection.query(
        query_embeddings=[embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    return [
        {
            "content": doc,
            "metadata": {
                "source": meta.get("source", ""),
                "page": meta.get("page", 0),
                "score": round(1 - dist, 4),
            },
        }
        for doc, meta, dist in zip(docs, metas, distances, strict=True)
    ]


def calculate(expression: str) -> str:
    """Avalia uma expressão matemática simples de forma segura.

    Usa o módulo ast para inspecionar a expressão antes de avaliá-la,
    garantindo que apenas operações aritméticas básicas sejam executadas.
    Qualquer tentativa de executar código arbitrário é rejeitada.

    Args:
        expression: expressão matemática em texto, ex: "2 + 3 * 4".

    Returns:
        Resultado numérico como string, ou mensagem de erro se inválida.
    """
    logger.debug("calculate: expression=%r", expression)
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _eval_node(tree.body)
        # Remove o .0 se for inteiro para melhor legibilidade
        if result == int(result):
            return str(int(result))
        return str(result)
    except (ValueError, ZeroDivisionError, SyntaxError, TypeError) as e:
        return f"Erro ao calcular: {e}"


def summarize(text: str) -> str:
    """Resume um texto longo usando o llama3.2 via Ollama.

    Útil para condensar trechos recuperados pelo search_papers antes
    de incluí-los no raciocínio do agente.

    Args:
        text: texto a ser resumido.

    Returns:
        Resumo curto gerado pelo LLM, ou mensagem de erro se falhar.
    """
    logger.debug("summarize: text length=%d", len(text))
    prompt = (
        "Resuma o seguinte texto em no máximo 3 frases, "
        "mantendo os pontos mais importantes:\n\n" + text
    )
    try:
        response = ollama.chat(
            model=GENERATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.message.content
    except Exception as e:  # noqa: BLE001
        return f"Erro ao resumir: {e}"
