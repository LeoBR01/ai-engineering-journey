# fase5-production/tests/test_monitor.py
import json
import logging
from unittest.mock import MagicMock, patch

from fastapi import BackgroundTasks

from src.monitor import EVAL_SAMPLE_RATE, _run_eval, maybe_schedule_eval


def test_eval_scheduled_when_random_below_rate() -> None:
    background = MagicMock(spec=BackgroundTasks)
    with patch("src.monitor.random.random", return_value=0.05):  # 0.05 < 0.1
        maybe_schedule_eval("q", "a", background, "abc123")
    background.add_task.assert_called_once()


def test_eval_not_scheduled_when_random_above_rate() -> None:
    background = MagicMock(spec=BackgroundTasks)
    with patch("src.monitor.random.random", return_value=0.99):  # 0.99 > 0.1
        maybe_schedule_eval("q", "a", background, "abc123")
    background.add_task.assert_not_called()


def test_eval_sample_rate_value() -> None:
    assert EVAL_SAMPLE_RATE == 0.1


def test_run_eval_logs_relevance(caplog) -> None:
    with (
        patch("src.monitor._answer_relevance", return_value=0.85),
        patch("src.monitor._FASE2_AVAILABLE", True),
        caplog.at_level(logging.INFO, logger="src.monitor"),
    ):
        _run_eval("What is RAG?", "RAG is a retrieval technique", "abc123")

    entries = [r for r in caplog.records if "answer_relevance" in r.message]
    assert len(entries) == 1
    entry = json.loads(entries[0].message)
    assert entry["trace_id"] == "abc123"
    assert entry["answer_relevance"] == 0.85


def test_run_eval_skips_when_fase2_unavailable(caplog) -> None:
    with (
        patch("src.monitor._FASE2_AVAILABLE", False),
        caplog.at_level(logging.WARNING, logger="src.monitor"),
    ):
        _run_eval("q", "a", "abc123")

    assert any("eval_skipped" in r.message for r in caplog.records)


def test_run_eval_handles_exception(caplog) -> None:
    with (
        patch("src.monitor._answer_relevance", side_effect=Exception("ollama down")),
        patch("src.monitor._FASE2_AVAILABLE", True),
        caplog.at_level(logging.WARNING, logger="src.monitor"),
    ):
        _run_eval("q", "a", "abc123")  # não deve levantar exceção

    assert any("eval_failed" in r.message for r in caplog.records)
