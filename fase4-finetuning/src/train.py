"""Treinamento QLoRA com Unsloth e TRL SFTTrainer.

Requer instalação manual do Unsloth:
    uv run pip install "unsloth[cu124-torch250]" trl peft

Uso:
    uv run train-model --dataset data/processed/train.jsonl \\
                       --output output/lora_adapter
"""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Importações no nível do módulo para permitir @patch("src.train.X") nos testes.
# try/except porque requerem GPU + instalação especial do Unsloth.
try:
    import torch  # noqa: F401
    from trl import SFTConfig, SFTTrainer  # noqa: F401
    from unsloth import FastLanguageModel  # noqa: F401
except ImportError:
    # Instalar com: uv run pip install "unsloth[cu124-torch250]" trl peft
    torch = None  # type: ignore[assignment]
    FastLanguageModel = None  # type: ignore[assignment]
    SFTConfig = None  # type: ignore[assignment]
    SFTTrainer = None  # type: ignore[assignment]


@dataclass
class TrainConfig:
    """Configuração completa do treinamento QLoRA."""

    model_name: str = "unsloth/llama-3.2-3b-instruct"
    dataset_path: str = "data/processed/train.jsonl"
    output_dir: str = "output/lora_adapter"
    max_seq_length: int = 2048
    load_in_4bit: bool = True

    # LoRA
    r: int = 16
    lora_alpha: int = 32  # scaling factor = lora_alpha / r = 2.0 (spec default)
    lora_dropout: float = 0.0

    # Treinamento
    batch_size: int = 4
    grad_accumulation: int = 4
    learning_rate: float = 2e-4
    num_epochs: int = 3
    warmup_steps: int = 5
    seed: int = 42


def load_and_format_dataset(path: str, tokenizer, max_seq_length: int = 2048):
    """Carrega JSONL ChatML, aplica chat template e tokeniza em processo único.

    Pre-tokeniza o dataset para evitar que SFTTrainer chame _prepare_dataset
    com multiprocessing — que falha porque ConfigModuleInstance do Unsloth
    não é serializável via pickle/dill.

    Args:
        path: caminho para arquivo .jsonl com exemplos em formato ChatML.
        tokenizer: tokenizer com método apply_chat_template.
        max_seq_length: tamanho máximo de sequência para truncamento.

    Returns:
        HuggingFace Dataset com colunas 'input_ids', 'attention_mask' e 'labels'.
    """
    from datasets import Dataset

    input_ids_list = []
    attention_mask_list = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            text = tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_seq_length,
                padding=False,
                return_tensors=None,
            )
            input_ids_list.append(encoded["input_ids"])
            attention_mask_list.append(encoded["attention_mask"])

    return Dataset.from_dict(
        {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "labels": input_ids_list,  # causal LM: labels == input_ids
        }
    )


def train(config: TrainConfig) -> Path:
    """Executa o treinamento QLoRA e salva o adapter LoRA.

    Args:
        config: TrainConfig com todos os hiperparâmetros.

    Returns:
        Path para o diretório onde o adapter foi salvo.
    """
    if FastLanguageModel is None:
        raise ImportError(
            "Unsloth não instalado. "
            "Execute: uv run pip install 'unsloth[cu124-torch250]' trl peft"
        )

    logger.info("Carregando modelo base: %s", config.model_name)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=None,
        load_in_4bit=config.load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=config.r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config.seed,
    )

    logger.info("Carregando dataset: %s", config.dataset_path)
    train_dataset = load_and_format_dataset(
        config.dataset_path, tokenizer, config.max_seq_length
    )
    logger.info("Dataset carregado: %d exemplos", len(train_dataset))

    # Verifica suporte a bf16 apenas se torch estiver disponível
    use_bf16 = bool(torch is not None and torch.cuda.is_bf16_supported())

    # Workaround: TRL>=0.12 valida placeholders do chat template (<EOS_TOKEN>,
    # <PAD_TOKEN> etc.) no vocab do Unsloth (TokenizersBackend), que não os expõe.
    # Adicionamos todos de uma vez para que a validação passe sem alterar o modelo.
    _placeholder_tokens = ["<EOS_TOKEN>", "<PAD_TOKEN>", "<BOS_TOKEN>", "<UNK_TOKEN>"]
    _missing = [t for t in _placeholder_tokens if t not in tokenizer.get_vocab()]
    if _missing:
        tokenizer.add_special_tokens({"additional_special_tokens": _missing})
        model.resize_token_embeddings(len(tokenizer))

    # Subclasse local que substitui compute_loss do TRL>=0.15 para evitar
    # entropy_from_logits — que falha porque Unsloth retorna outputs.logits
    # como função (lazy), não como tensor.
    class _Trainer(SFTTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            outputs = model(**inputs)
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

    # Dataset já pre-tokenizado em load_and_format_dataset — SFTTrainer
    # pula _prepare_dataset quando encontra coluna 'input_ids' no dataset.
    trainer = _Trainer(
        model=model,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        args=SFTConfig(
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.grad_accumulation,
            warmup_steps=config.warmup_steps,
            num_train_epochs=config.num_epochs,
            learning_rate=config.learning_rate,
            fp16=not use_bf16,
            bf16=use_bf16,
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=config.seed,
            output_dir=config.output_dir,
        ),
    )

    logger.info("Iniciando treinamento...")
    trainer.train()

    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    logger.info("Adapter salvo em: %s", output_path)

    return output_path


def main() -> None:
    """CLI para treinamento QLoRA."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Treinamento QLoRA com Unsloth")
    parser.add_argument("--dataset", default="data/processed/train.jsonl")
    parser.add_argument("--output", default="output/lora_adapter")
    parser.add_argument("--model", default="unsloth/llama-3.2-3b-instruct")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16)
    args = parser.parse_args()

    config = TrainConfig(
        model_name=args.model,
        dataset_path=args.dataset,
        output_dir=args.output,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        r=args.rank,
    )
    output_path = train(config)
    print(f"Treinamento concluído. Adapter em: {output_path}")


if __name__ == "__main__":
    main()
