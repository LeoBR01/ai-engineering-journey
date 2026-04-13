"""Testes unitários para tools.py — totalmente mockados, sem ChromaDB nem Ollama."""

from unittest.mock import MagicMock, patch

import pytest

from src.tools import calculate, search_papers, summarize

# ── Fixtures ──────────────────────────────────────────────────────────────────

MOCK_CHROMA_RESULTS = {
    "documents": [["RAG combina retrieval com geração.", "Chunking divide docs."]],
    "metadatas": [[{"source": "rag.pdf", "page": 1}, {"source": "rag.pdf", "page": 2}]],
    "distances": [[0.1, 0.3]],
}


# ── search_papers ─────────────────────────────────────────────────────────────


@patch("src.tools.ollama.embed", return_value=MagicMock(embeddings=[[0.1] * 768]))
@patch("src.tools.chromadb.PersistentClient")
def test_search_papers_retorna_lista_com_campos_corretos(mock_client_cls, mock_embed):
    """search_papers deve retornar lista de dicts com 'content' e 'metadata'."""
    mock_collection = MagicMock()
    mock_collection.count.return_value = 50
    mock_collection.query.return_value = MOCK_CHROMA_RESULTS
    mock_client_cls.return_value.get_collection.return_value = mock_collection

    results = search_papers("O que é RAG?", k=2)

    assert isinstance(results, list)
    assert len(results) == 2
    assert results[0]["content"] == "RAG combina retrieval com geração."
    assert results[0]["metadata"]["source"] == "rag.pdf"
    assert results[0]["metadata"]["page"] == 1
    assert results[0]["metadata"]["score"] == pytest.approx(0.9)


@patch("src.tools.ollama.embed", return_value=MagicMock(embeddings=[[0.1] * 768]))
@patch("src.tools.chromadb.PersistentClient")
def test_search_papers_colecao_vazia_retorna_lista_vazia(mock_client_cls, mock_embed):
    """search_papers deve retornar [] sem chamar query se a collection estiver vazia."""
    mock_collection = MagicMock()
    mock_collection.count.return_value = 0
    mock_client_cls.return_value.get_collection.return_value = mock_collection

    results = search_papers("qualquer coisa", k=5)

    assert results == []
    mock_collection.query.assert_not_called()


@patch("src.tools.ollama.embed", return_value=MagicMock(embeddings=[[0.1] * 768]))
@patch("src.tools.chromadb.PersistentClient")
def test_search_papers_respeita_parametro_k(mock_client_cls, mock_embed):
    """search_papers deve solicitar min(k, count) resultados ao ChromaDB."""
    mock_collection = MagicMock()
    mock_collection.count.return_value = 3  # menos que k=5
    mock_collection.query.return_value = {
        "documents": [["a", "b", "c"]],
        "metadatas": [[{"source": "x.pdf", "page": 1}] * 3],
        "distances": [[0.1, 0.2, 0.3]],
    }
    mock_client_cls.return_value.get_collection.return_value = mock_collection

    results = search_papers("query", k=5)

    call_kwargs = mock_collection.query.call_args[1]
    assert call_kwargs["n_results"] == 3
    assert len(results) == 3


# ── calculate ─────────────────────────────────────────────────────────────────


def test_calculate_soma():
    """calculate deve somar dois números corretamente."""
    assert calculate("2 + 3") == "5"


def test_calculate_multiplicacao_e_precedencia():
    """calculate deve respeitar precedência de operadores."""
    assert calculate("2 + 3 * 4") == "14"


def test_calculate_divisao_retorna_float():
    """calculate deve retornar float quando a divisão não é exata."""
    result = calculate("7 / 2")
    assert result == "3.5"


def test_calculate_divisao_inteira():
    """calculate deve retornar inteiro para divisão exata."""
    assert calculate("10 / 2") == "5"


def test_calculate_potencia():
    """calculate deve suportar o operador de potência."""
    assert calculate("2 ** 8") == "256"


def test_calculate_expressao_com_parenteses():
    """calculate deve respeitar parênteses."""
    assert calculate("(2 + 3) * 4") == "20"


def test_calculate_divisao_por_zero_retorna_erro():
    """calculate deve retornar mensagem de erro para divisão por zero."""
    result = calculate("10 / 0")
    assert result.startswith("Erro ao calcular:")


def test_calculate_expressao_invalida_retorna_erro():
    """calculate deve retornar mensagem de erro para expressão inválida."""
    result = calculate("abc + 1")
    assert result.startswith("Erro ao calcular:")


def test_calculate_rejeita_chamada_de_funcao():
    """calculate deve rejeitar qualquer chamada de função para segurança."""
    result = calculate("__import__('os').system('echo pwned')")
    assert result.startswith("Erro ao calcular:")


def test_calculate_rejeita_atributo():
    """calculate deve rejeitar acesso a atributos (ex: os.path)."""
    result = calculate("(1).__class__")
    assert result.startswith("Erro ao calcular:")


def test_calculate_rejeita_string_literal():
    """calculate deve rejeitar strings — só aceita números."""
    result = calculate('"hello"')
    assert result.startswith("Erro ao calcular:")


def test_calculate_negativo():
    """calculate deve suportar números negativos via UnaryOp."""
    assert calculate("-5 + 10") == "5"


# ── summarize ─────────────────────────────────────────────────────────────────


@patch("src.tools.ollama.chat")
def test_summarize_retorna_string(mock_chat):
    """summarize deve retornar a string gerada pelo LLM."""
    mock_response = MagicMock()
    mock_response.message.content = "Resumo: RAG combina busca com geração."
    mock_chat.return_value = mock_response

    result = summarize("Texto longo sobre RAG...")

    assert isinstance(result, str)
    assert result == "Resumo: RAG combina busca com geração."


@patch("src.tools.ollama.chat")
def test_summarize_chama_ollama_com_modelo_correto(mock_chat):
    """summarize deve chamar ollama.chat com o modelo configurado."""
    mock_response = MagicMock()
    mock_response.message.content = "Resumo."
    mock_chat.return_value = mock_response

    summarize("Texto qualquer")

    call_kwargs = mock_chat.call_args[1]
    assert call_kwargs["model"] == "llama3.2"


@patch("src.tools.ollama.chat")
def test_summarize_inclui_texto_no_prompt(mock_chat):
    """summarize deve incluir o texto original na mensagem enviada ao LLM."""
    mock_response = MagicMock()
    mock_response.message.content = "Resumo."
    mock_chat.return_value = mock_response

    texto = "Conteúdo específico a ser resumido."
    summarize(texto)

    messages = mock_chat.call_args[1]["messages"]
    assert any(texto in m["content"] for m in messages)


@patch("src.tools.ollama.chat", side_effect=ConnectionError("Ollama offline"))
def test_summarize_retorna_erro_quando_ollama_falha(mock_chat):
    """summarize deve retornar mensagem de erro se o Ollama não responder."""
    result = summarize("Qualquer texto")

    assert result.startswith("Erro ao resumir:")
