"""Testes para o módulo de avaliação comparativa."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.evaluate import _compare_results, _load_latest_baseline, make_generate_fn


class TestCompareResults:
    def test_delta_calculated_correctly(self) -> None:
        current = {
            "avg_faithfulness": 0.75,
            "avg_answer_relevance": 0.72,
        }
        baseline = {
            "avg_faithfulness": 0.57,
            "avg_answer_relevance": 0.60,
        }

        result = _compare_results(current, baseline, model_name="llama3.2-finetuned")

        faith = result["metrics"]["faithfulness"]
        assert abs(faith["delta"] - 0.18) < 0.001
        assert faith["baseline"] == 0.57
        assert faith["finetuned"] == 0.75
        assert faith["achieved"] is True  # 0.75 > 0.70

    def test_target_not_achieved_when_below_threshold(self) -> None:
        current = {"avg_faithfulness": 0.65, "avg_answer_relevance": 0.65}
        baseline = {"avg_faithfulness": 0.57, "avg_answer_relevance": 0.60}

        result = _compare_results(current, baseline, model_name="llama3.2-finetuned")

        assert result["metrics"]["faithfulness"]["achieved"] is False

    def test_model_name_in_result(self) -> None:
        current = {"avg_faithfulness": 0.75, "avg_answer_relevance": 0.72}
        baseline = {"avg_faithfulness": 0.57, "avg_answer_relevance": 0.60}

        result = _compare_results(current, baseline, model_name="my-model")

        assert result["model"] == "my-model"

    def test_timestamp_in_result(self) -> None:
        current = {"avg_faithfulness": 0.75, "avg_answer_relevance": 0.72}
        baseline = {"avg_faithfulness": 0.57, "avg_answer_relevance": 0.60}

        result = _compare_results(current, baseline, model_name="model")

        assert "timestamp" in result


class TestLoadLatestBaseline:
    def test_loads_most_recent_json(self) -> None:
        baseline_data = {"avg_faithfulness": 0.57, "avg_answer_relevance": 0.60}

        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_dir = Path(tmpdir)
            (baseline_dir / "eval_fase1_20260411_192100.json").write_text(
                json.dumps(baseline_data)
            )

            result = _load_latest_baseline(str(baseline_dir))

        assert result["avg_faithfulness"] == 0.57

    def test_raises_if_no_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, pytest.raises(FileNotFoundError):
            _load_latest_baseline(tmpdir)

    def test_loads_alphabetically_last_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_dir = Path(tmpdir)
            (baseline_dir / "eval_fase1_20260411_120000.json").write_text(
                json.dumps({"avg_faithfulness": 0.50})
            )
            (baseline_dir / "eval_fase1_20260411_192100.json").write_text(
                json.dumps({"avg_faithfulness": 0.57})
            )

            result = _load_latest_baseline(str(baseline_dir))

        assert result["avg_faithfulness"] == 0.57


class TestMakeGenerateFn:
    @patch("src.evaluate.ollama")
    def test_uses_correct_model(self, mock_ollama: MagicMock) -> None:
        mock_ollama.chat.return_value = MagicMock(message=MagicMock(content="Answer."))

        generate_fn = make_generate_fn("llama3.2-finetuned")
        result = generate_fn("Question?", "Some context.")

        call_kwargs = mock_ollama.chat.call_args
        assert call_kwargs.kwargs["model"] == "llama3.2-finetuned"
        assert result == "Answer."

    @patch("src.evaluate.ollama")
    def test_context_in_user_message(self, mock_ollama: MagicMock) -> None:
        mock_ollama.chat.return_value = MagicMock(message=MagicMock(content="A."))

        generate_fn = make_generate_fn("model")
        generate_fn("Q?", "unique_context_xyz")

        call_args = mock_ollama.chat.call_args
        messages = call_args.kwargs["messages"]
        # Contexto vai na mensagem user (não no system) para manter
        # consistência com o formato do dataset de treino.
        user_content = next(m["content"] for m in messages if m["role"] == "user")
        assert "unique_context_xyz" in user_content
