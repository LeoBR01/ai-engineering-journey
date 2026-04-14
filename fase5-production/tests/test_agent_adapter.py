"""Testes para o AgentAdapter que encapsula o ReActAgent da Fase 3."""

from unittest.mock import MagicMock

from src.agent_adapter import AgentAdapter


def _make_adapter() -> AgentAdapter:
    """Cria AgentAdapter sem instanciar ReActAgent."""
    adapter = AgentAdapter.__new__(AgentAdapter)
    adapter._agent = MagicMock()
    return adapter


def test_run_returns_answer_string() -> None:
    adapter = _make_adapter()
    adapter._agent.run.return_value = MagicMock(answer="agent answer")
    result = adapter.run("Compare RAG and fine-tuning")
    assert result == "agent answer"


def test_run_passes_question_to_agent() -> None:
    adapter = _make_adapter()
    adapter._agent.run.return_value = MagicMock(answer="answer")
    adapter.run("the question")
    adapter._agent.run.assert_called_once_with("the question")


def test_run_returns_string_type() -> None:
    adapter = _make_adapter()
    adapter._agent.run.return_value = MagicMock(answer="response text")
    result = adapter.run("question")
    assert isinstance(result, str)
