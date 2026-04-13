"""Testes para o módulo de treinamento QLoRA."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.train import TrainConfig, load_and_format_dataset


class TestTrainConfig:
    def test_default_values(self) -> None:
        config = TrainConfig()
        assert config.model_name == "unsloth/llama-3.2-3b-instruct"
        assert config.r == 16
        assert config.lora_alpha == 32
        assert config.load_in_4bit is True
        assert config.num_epochs == 3
        assert config.learning_rate == 2e-4
        assert config.batch_size == 4
        assert config.grad_accumulation == 4
        assert config.max_seq_length == 2048
        assert config.seed == 42

    def test_custom_values(self) -> None:
        config = TrainConfig(r=32, num_epochs=5, learning_rate=1e-4)
        assert config.r == 32
        assert config.num_epochs == 5
        assert config.learning_rate == 1e-4


class TestLoadAndFormatDataset:
    def test_loads_jsonl_and_applies_template(self) -> None:
        examples = [
            {
                "messages": [
                    {"role": "system", "content": "System prompt."},
                    {"role": "user", "content": "Question?"},
                    {"role": "assistant", "content": "Answer."},
                ]
            }
        ] * 5

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = (
            "<|system|>System<|user|>Q<|assistant|>A"
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
            tmp_path = f.name

        dataset = load_and_format_dataset(tmp_path, mock_tokenizer)

        assert len(dataset) == 5
        assert "text" in dataset.column_names
        assert mock_tokenizer.apply_chat_template.call_count == 5

    def test_apply_chat_template_called_correctly(self) -> None:
        example = {
            "messages": [
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": "A"},
            ]
        }

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "formatted text"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(example) + "\n")
            tmp_path = f.name

        load_and_format_dataset(tmp_path, mock_tokenizer)

        mock_tokenizer.apply_chat_template.assert_called_once_with(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )

    def test_skips_empty_lines(self) -> None:
        example = {
            "messages": [
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": "A"},
            ]
        }

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "text"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(example) + "\n")
            f.write("\n")  # empty line
            f.write(json.dumps(example) + "\n")
            tmp_path = f.name

        dataset = load_and_format_dataset(tmp_path, mock_tokenizer)
        assert len(dataset) == 2


class TestTrainFunction:
    @patch("src.train.SFTTrainer")
    @patch("src.train.TrainingArguments")
    @patch("src.train.FastLanguageModel")
    def test_train_calls_trainer_train(
        self, mock_flm: MagicMock, mock_args: MagicMock, mock_trainer_cls: MagicMock
    ) -> None:
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "text"
        mock_flm.from_pretrained.return_value = (mock_model, mock_tokenizer)
        mock_flm.get_peft_model.return_value = mock_model

        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "train.jsonl"
            example = {
                "messages": [
                    {"role": "user", "content": "Q"},
                    {"role": "assistant", "content": "A"},
                ]
            }
            dataset_path.write_text(json.dumps(example) + "\n")

            output_dir = Path(tmpdir) / "adapter"
            from src.train import TrainConfig, train

            config = TrainConfig(
                dataset_path=str(dataset_path),
                output_dir=str(output_dir),
            )
            train(config)

        mock_trainer.train.assert_called_once()
        mock_model.save_pretrained.assert_called_once()
        mock_tokenizer.save_pretrained.assert_called_once()

    @patch("src.train.SFTTrainer")
    @patch("src.train.TrainingArguments")
    @patch("src.train.FastLanguageModel")
    def test_train_returns_output_path(
        self, mock_flm: MagicMock, mock_args: MagicMock, mock_trainer_cls: MagicMock
    ) -> None:
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "text"
        mock_flm.from_pretrained.return_value = (mock_model, mock_tokenizer)
        mock_flm.get_peft_model.return_value = mock_model
        mock_trainer_cls.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "train.jsonl"
            example = {
                "messages": [
                    {"role": "user", "content": "Q"},
                    {"role": "assistant", "content": "A"},
                ]
            }
            dataset_path.write_text(json.dumps(example) + "\n")

            output_dir = Path(tmpdir) / "adapter"
            from src.train import TrainConfig, train

            config = TrainConfig(
                dataset_path=str(dataset_path),
                output_dir=str(output_dir),
            )
            result = train(config)

        assert result == output_dir
