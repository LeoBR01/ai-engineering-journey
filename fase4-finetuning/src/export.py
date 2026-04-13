"""Export do adapter LoRA para GGUF e registro no Ollama.

Uso:
    uv run export-model --adapter output/lora_adapter \\
                        --gguf-output output/model_gguf \\
                        --model-name llama3.2-finetuned
"""

import argparse
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# Importação no nível do módulo para permitir
# @patch("src.export.FastLanguageModel") nos testes.
try:
    from unsloth import FastLanguageModel  # noqa: F401
except ImportError:
    FastLanguageModel = None  # type: ignore[assignment]


def export_to_gguf(
    adapter_path: str,
    output_gguf_path: str,
    quantization: str = "q4_k_m",
) -> Path:
    """Converte o adapter LoRA para formato GGUF via Unsloth.

    Carrega o adapter LoRA, mergeia com o modelo base e exporta para GGUF
    com quantização Q4_K_M (balanço entre qualidade e tamanho).

    Args:
        adapter_path: caminho para o diretório com o adapter salvo pelo Unsloth.
        output_gguf_path: diretório de destino para o arquivo GGUF.
        quantization: método de quantização (q4_k_m, q8_0, f16).

    Returns:
        Path para o diretório com o GGUF gerado.
    """
    if FastLanguageModel is None:
        raise ImportError(
            "Unsloth não instalado. "
            "Execute: uv run pip install 'unsloth[cu124-torch250]' trl peft"
        )

    logger.info("Carregando adapter: %s", adapter_path)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    output_path = Path(output_gguf_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # save_pretrained_gguf realiza o merge adapter+base internamente —
    # dispensa save_pretrained_merged explícito + conversão via llama.cpp.
    # O checkpoint merged intermediário (output/merged/) não é gerado em disco.
    logger.info("Exportando para GGUF (%s)...", quantization)
    model.save_pretrained_gguf(
        str(output_path),
        tokenizer,
        quantization_method=quantization,
    )
    logger.info("GGUF salvo em: %s", output_path)

    return output_path


def create_modelfile(
    gguf_dir: str,
    model_name: str = "llama3.2-finetuned",
) -> Path:
    """Cria o Modelfile para registro no Ollama.

    Args:
        gguf_dir: diretório contendo o arquivo .gguf.
        model_name: nome do modelo no Ollama (para referência).

    Returns:
        Path para o Modelfile criado.

    Raises:
        FileNotFoundError: se nenhum arquivo .gguf for encontrado em gguf_dir.
    """
    gguf_path = Path(gguf_dir)
    gguf_files = sorted(gguf_path.glob("*.gguf"))

    if not gguf_files:
        raise FileNotFoundError(f"Nenhum arquivo .gguf encontrado em: {gguf_dir}")

    gguf_file = gguf_files[0]
    modelfile_content = (
        f"FROM ./{gguf_file.name}\nPARAMETER temperature 0.7\nPARAMETER num_ctx 4096\n"
    )

    modelfile_path = gguf_path / "Modelfile"
    modelfile_path.write_text(modelfile_content, encoding="utf-8")
    logger.info("Modelfile criado: %s", modelfile_path)

    return modelfile_path


def register_with_ollama(
    gguf_dir: str,
    model_name: str = "llama3.2-finetuned",
) -> str:
    """Registra o modelo GGUF no Ollama via CLI.

    Args:
        gguf_dir: diretório com o .gguf e o Modelfile.
        model_name: nome para registrar no Ollama.

    Returns:
        Nome do modelo registrado.

    Raises:
        subprocess.CalledProcessError: se o ollama create falhar.
    """
    modelfile_path = create_modelfile(gguf_dir, model_name)

    logger.info("Registrando no Ollama como: %s", model_name)
    subprocess.run(
        ["ollama", "create", model_name, "-f", str(modelfile_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    logger.info("Modelo registrado: %s", model_name)

    return model_name


def export_and_register(
    adapter_path: str = "output/lora_adapter",
    gguf_output_path: str = "output/model_gguf",
    model_name: str = "llama3.2-finetuned",
    quantization: str = "q4_k_m",
) -> str:
    """Pipeline completo: adapter → GGUF → Ollama.

    Args:
        adapter_path: diretório do adapter LoRA.
        gguf_output_path: diretório de destino para o GGUF.
        model_name: nome do modelo no Ollama.
        quantization: método de quantização GGUF.

    Returns:
        Nome do modelo registrado no Ollama.
    """
    export_to_gguf(adapter_path, gguf_output_path, quantization)
    return register_with_ollama(gguf_output_path, model_name)


def main() -> None:
    """CLI para export e registro do modelo."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Exporta adapter LoRA para Ollama")
    parser.add_argument("--adapter", default="output/lora_adapter")
    parser.add_argument("--gguf-output", default="output/model_gguf")
    parser.add_argument("--model-name", default="llama3.2-finetuned")
    parser.add_argument("--quantization", default="q4_k_m")
    args = parser.parse_args()

    model_name = export_and_register(
        args.adapter, args.gguf_output, args.model_name, args.quantization
    )
    print(f"Modelo disponível no Ollama: {model_name}")
    print(f"Teste com: ollama run {model_name} 'Olá, funciona?'")


if __name__ == "__main__":
    main()
