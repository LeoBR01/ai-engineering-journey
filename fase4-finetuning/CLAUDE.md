# CLAUDE.md — Fase 4: Fine-tuning

## Visão Geral

Fine-tuning supervisionado (SFT) com QLoRA no llama3.2 3B para melhorar
qualidade RAG (Faithfulness, Answer Relevance) e aderência ao formato ReAct.

## Dependências de Outras Fases

- **Fase 1** (`../fase1-rag/`): ChromaDB com 4.110 chunks, usado para gerar o dataset RAG
- **Fase 2** (`../fase2-evals/`): sistema de evals, reutilizado para medir melhoria
- **Fase 3** (`../fase3-agents/`): SYSTEM_PROMPT e search_papers, usados para gerar ciclos ReAct

## Gerenciamento de Ambiente

```bash
# Instalar dependências base
uv sync

# Instalar Unsloth + TRL (escolher string correta para CUDA)
# Para CUDA 12.4 + PyTorch 2.5.0:
uv run pip install "unsloth[cu124-torch250]" trl peft

# Verificar instalação
uv run python -c "from unsloth import FastLanguageModel; print('OK')"
```

## Fluxo Completo

```bash
# 1. Gerar dataset (precisa de Ollama + ChromaDB da Fase 1)
uv run generate-dataset --rag-output data/raw/rag_pairs.jsonl \
                        --react-output data/raw/react_cycles.jsonl \
                        --processed-output data/processed/train.jsonl

# 2. Treinar (precisa de GPU ≥8GB VRAM)
uv run train-model --dataset data/processed/train.jsonl \
                   --output output/lora_adapter

# 3. Exportar para Ollama
uv run export-model --adapter output/lora_adapter \
                    --gguf-output output/model_gguf \
                    --model-name llama3.2-finetuned

# 4. Avaliar
uv run evaluate-model --model llama3.2-finetuned \
                      --output data/results
```

## Lint e Formatação

```bash
ruff check . && ruff format .
```

## Estrutura de Módulos

| Arquivo | Responsabilidade |
|---|---|
| `generate_dataset.py` | Geração de pares RAG e ciclos ReAct + preprocessamento |
| `train.py` | `TrainConfig` + loop de treinamento QLoRA com Unsloth |
| `export.py` | adapter → GGUF → registro no Ollama |
| `evaluate.py` | Wrapper sobre fase2-evals com modelo parametrizado |
