"""Testes para o módulo de export GGUF e registro no Ollama."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.export import create_modelfile, register_with_ollama


class TestCreateModelfile:
    def test_creates_modelfile_in_gguf_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_dir = Path(tmpdir)
            (gguf_dir / "model.gguf").touch()

            modelfile_path = create_modelfile(str(gguf_dir), "llama3.2-finetuned")

            assert modelfile_path.exists()
            assert modelfile_path.name == "Modelfile"
            assert modelfile_path.parent == gguf_dir

    def test_modelfile_contains_from_directive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_dir = Path(tmpdir)
            (gguf_dir / "model.gguf").touch()

            modelfile_path = create_modelfile(str(gguf_dir))

            content = modelfile_path.read_text()
            assert "FROM ./model.gguf" in content

    def test_modelfile_contains_temperature(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_dir = Path(tmpdir)
            (gguf_dir / "model.gguf").touch()

            modelfile_path = create_modelfile(str(gguf_dir))

            content = modelfile_path.read_text()
            assert "temperature" in content

    def test_raises_if_no_gguf_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, pytest.raises(FileNotFoundError):
            create_modelfile(tmpdir)


class TestRegisterWithOllama:
    @patch("src.export.subprocess.run")
    def test_calls_ollama_create(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_dir = Path(tmpdir)
            (gguf_dir / "model.gguf").touch()

            register_with_ollama(str(gguf_dir), "llama3.2-finetuned")

        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "ollama" in call_args
        assert "create" in call_args
        assert "llama3.2-finetuned" in call_args

    @patch("src.export.subprocess.run")
    def test_returns_model_name(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_dir = Path(tmpdir)
            (gguf_dir / "model.gguf").touch()

            result = register_with_ollama(str(gguf_dir), "llama3.2-finetuned")

        assert result == "llama3.2-finetuned"

    @patch("src.export.subprocess.run")
    def test_passes_modelfile_path(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_dir = Path(tmpdir)
            (gguf_dir / "model.gguf").touch()

            register_with_ollama(str(gguf_dir), "my-model")

        call_args = mock_run.call_args[0][0]
        assert "-f" in call_args


class TestExportToGguf:
    @patch("src.export.FastLanguageModel")
    def test_calls_save_pretrained_gguf(self, mock_flm: MagicMock) -> None:
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_flm.from_pretrained.return_value = (mock_model, mock_tokenizer)

        with tempfile.TemporaryDirectory() as tmpdir:
            from src.export import export_to_gguf

            export_to_gguf(str(Path(tmpdir) / "adapter"), str(Path(tmpdir) / "gguf"))

        mock_model.save_pretrained_gguf.assert_called_once()
        call_kwargs = mock_model.save_pretrained_gguf.call_args
        assert "quantization_method" in call_kwargs.kwargs or len(call_kwargs.args) >= 3
