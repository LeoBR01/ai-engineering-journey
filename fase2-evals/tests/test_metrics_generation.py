"""Testes unitários para as métricas de geração (LLM-as-judge)."""

from unittest.mock import MagicMock, patch

from src.metrics_generation import _parse_score, answer_relevance, faithfulness

# ── _parse_score ──


def test_parse_score_numero_exato():
    assert _parse_score("0.85") == 0.85


def test_parse_score_inteiro():
    assert _parse_score("1") == 1.0


def test_parse_score_com_texto_extra():
    assert _parse_score("Score: 0.7 (bom)") == 0.7


def test_parse_score_invalido_retorna_zero():
    assert _parse_score("sem número") == 0.0


def test_parse_score_clamp_maximo():
    assert _parse_score("1.5") == 1.0


def test_parse_score_clamp_minimo():
    assert _parse_score("-0.3") == 0.0


# ── faithfulness ──


def _mock_ollama_response(score: str) -> MagicMock:
    mock = MagicMock()
    mock.__getitem__ = lambda self, key: {"message": {"content": score}}[key]
    return mock


@patch("src.metrics_generation.ollama.chat")
def test_faithfulness_score_alto(mock_chat):
    mock_chat.return_value = {"message": {"content": "1.0"}}
    score = faithfulness("A resposta correta.", "Contexto que suporta a resposta.")
    assert score == 1.0
    mock_chat.assert_called_once()


@patch("src.metrics_generation.ollama.chat")
def test_faithfulness_score_baixo(mock_chat):
    mock_chat.return_value = {"message": {"content": "0.1"}}
    score = faithfulness("Resposta inventada.", "Contexto não relacionado.")
    assert score == 0.1


@patch("src.metrics_generation.ollama.chat")
def test_faithfulness_usa_modelo_correto(mock_chat):
    mock_chat.return_value = {"message": {"content": "0.5"}}
    faithfulness("resposta", "contexto", model="llama3.2")
    call_kwargs = mock_chat.call_args
    assert call_kwargs[1]["model"] == "llama3.2" or call_kwargs[0][0] == "llama3.2"


# ── answer_relevance ──


@patch("src.metrics_generation.ollama.chat")
def test_answer_relevance_score_alto(mock_chat):
    mock_chat.return_value = {"message": {"content": "0.9"}}
    score = answer_relevance("Resposta relevante.", "Qual é a capital do Brasil?")
    assert score == 0.9


@patch("src.metrics_generation.ollama.chat")
def test_answer_relevance_score_baixo(mock_chat):
    mock_chat.return_value = {"message": {"content": "0.0"}}
    score = answer_relevance("Resposta sobre outra coisa.", "Qual é a capital?")
    assert score == 0.0


@patch("src.metrics_generation.ollama.chat")
def test_answer_relevance_parse_texto_extra(mock_chat):
    mock_chat.return_value = {"message": {"content": "A pontuação é 0.75 de 1.0"}}
    score = answer_relevance("resposta", "pergunta")
    assert score == 0.75
