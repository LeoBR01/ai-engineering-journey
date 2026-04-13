# Fase 4: Fine-tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune llama3.2 3B com QLoRA para melhorar Faithfulness (0.57→>0.70), Answer Relevance (0.60→>0.70) e aderência ao formato ReAct (~70%→>90%).

**Architecture:** Dataset sintético misto (RAG pairs + ciclos ReAct) gerado do ChromaDB da Fase 1, treinamento QLoRA via Unsloth, export GGUF para Ollama, avaliação via sistema da Fase 2.

**Tech Stack:** Python 3.11+, uv, ruff, unsloth, trl (SFTTrainer), peft, transformers, torch, chromadb, ollama, pytest

---

## File Structure

```
fase4-finetuning/
├── CLAUDE.md
├── pyproject.toml
├── data/
│   ├── raw/
│   │   ├── rag_pairs.jsonl          # ~500-800 pares (context, pergunta, resposta)
│   │   └── react_cycles.jsonl       # ~200-300 ciclos ReAct completos
│   └── processed/
│       ├── train.jsonl              # dataset final merged, ChatML
│       └── val.jsonl                # 10% split de validação
├── output/
│   ├── lora_adapter/                # adapter LoRA salvo pelo Unsloth
│   └── model_gguf/                  # GGUF exportado + Modelfile
├── src/
│   ├── __init__.py
│   ├── generate_dataset.py          # RAG pairs + ReAct cycles + preprocess
│   ├── train.py                     # TrainConfig + QLoRA training
│   ├── export.py                    # adapter → GGUF → ollama create
│   └── evaluate.py                  # wrapper sobre fase2-evals
└── tests/
    ├── __init__.py
    ├── test_generate_dataset.py
    ├── test_train.py
    └── test_export.py
```

**Responsabilidades:**
- `generate_dataset.py` — toda lógica de geração e preprocessamento de dados de treino
- `train.py` — `TrainConfig` dataclass + funções de treino com Unsloth/TRL
- `export.py` — conversão adapter→GGUF e registro no Ollama
- `evaluate.py` — reutiliza `evaluate_dataset` da Fase 2 com modelo parametrizado

---

## Task 1: Scaffolding

**Files:**
- Create: `fase4-finetuning/pyproject.toml`
- Create: `fase4-finetuning/CLAUDE.md`
- Create: `fase4-finetuning/src/__init__.py`
- Create: `fase4-finetuning/tests/__init__.py`
- Create: `fase4-finetuning/data/raw/.gitkeep`
- Create: `fase4-finetuning/data/processed/.gitkeep`
- Create: `fase4-finetuning/output/.gitkeep`

- [ ] **Step 1: Criar estrutura de diretórios**

```bash
mkdir -p fase4-finetuning/src fase4-finetuning/tests \
         fase4-finetuning/data/raw fase4-finetuning/data/processed \
         fase4-finetuning/output
touch fase4-finetuning/data/raw/.gitkeep \
      fase4-finetuning/data/processed/.gitkeep \
      fase4-finetuning/output/.gitkeep \
      fase4-finetuning/src/__init__.py \
      fase4-finetuning/tests/__init__.py
```

- [ ] **Step 2: Criar `fase4-finetuning/pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src"]

[project]
name = "fase4-finetuning"
version = "0.1.0"
description = "Fine-tuning supervisionado com QLoRA para melhorar qualidade RAG"
requires-python = ">=3.11"
dependencies = [
    "chromadb>=0.5.0",
    "ollama>=0.3.0",
    "datasets>=3.0",
    "pytest>=8.0.0",
    "pytest-cov>=5.0.0",
]

[project.scripts]
generate-dataset = "src.generate_dataset:main"
train-model = "src.train:main"
export-model = "src.export:main"
evaluate-model = "src.evaluate:main"

[dependency-groups]
dev = [
    "ruff>=0.15.10",
]

[tool.ruff]
line-length = 88
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]
ignore = []

[tool.ruff.format]
quote-style = "double"
indent-style = "space"

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["integration: requer Ollama, ChromaDB e GPU rodando"]
```

> **Nota:** `unsloth`, `trl`, `peft`, `transformers`, `torch` não estão no pyproject.toml
> porque requerem instalação manual com versão CUDA correta.
> Após `uv sync`, instale via: `uv run pip install "unsloth[cu124-torch250]" trl peft`
> Veja https://github.com/unslothai/unsloth para a string correta para sua GPU.

- [ ] **Step 3: Criar `fase4-finetuning/CLAUDE.md`**

```markdown
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
```

- [ ] **Step 4: Inicializar ambiente uv**

```bash
cd fase4-finetuning && uv sync
```

Expected: `Resolved N packages` sem erros.

- [ ] **Step 5: Commit**

```bash
git add fase4-finetuning/
git commit -m "chore: scaffold fase4-finetuning — estrutura, pyproject, CLAUDE"
```

---

## Task 2: generate_dataset.py — RAG pairs (TDD)

**Files:**
- Create: `fase4-finetuning/src/generate_dataset.py`
- Create: `fase4-finetuning/tests/test_generate_dataset.py`

- [ ] **Step 1: Escrever testes para RAG pairs**

Criar `fase4-finetuning/tests/test_generate_dataset.py`:

```python
"""Testes para geração de dataset de fine-tuning."""

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from src.generate_dataset import (
    RagPair,
    filter_rag_pair,
    format_observation,
    generate_rag_pair,
    pair_to_chatml_rag,
)


class TestRagPairFilter:
    def test_valid_pair_passes(self) -> None:
        pair = RagPair(
            context="Some context text about machine learning.",
            question="What is machine learning?",
            answer="Machine learning is a method of data analysis that automates model building.",
        )
        assert filter_rag_pair(pair) is True

    def test_short_answer_fails(self) -> None:
        pair = RagPair(
            context="Context text here.",
            question="What is it?",
            answer="Short.",
        )
        assert filter_rag_pair(pair) is False

    def test_short_question_fails(self) -> None:
        pair = RagPair(
            context="Context text here.",
            question="Why?",
            answer="Because this is a long enough answer that passes the minimum length check.",
        )
        assert filter_rag_pair(pair) is False


class TestGenerateRagPair:
    def test_valid_response_parsed_correctly(self) -> None:
        def llm_fn(prompt: str) -> str:
            return "QUESTION: What are transformers used for?\nANSWER: Transformers are used for natural language processing tasks such as translation and summarization."

        chunk = "Transformers are a deep learning architecture used in NLP."
        pair = generate_rag_pair(chunk, llm_fn)

        assert pair is not None
        assert pair.context == chunk
        assert pair.question == "What are transformers used for?"
        assert "Transformers are used for" in pair.answer

    def test_malformed_response_returns_none(self) -> None:
        def llm_fn(prompt: str) -> str:
            return "This is not in the expected format."

        pair = generate_rag_pair("Some chunk text.", llm_fn)
        assert pair is None

    def test_prompt_includes_chunk_text(self) -> None:
        captured_prompts = []

        def llm_fn(prompt: str) -> str:
            captured_prompts.append(prompt)
            return "QUESTION: Test question here long enough?\nANSWER: Test answer here that is long enough to pass the filter check."

        generate_rag_pair("unique_chunk_marker_xyz", llm_fn)
        assert len(captured_prompts) == 1
        assert "unique_chunk_marker_xyz" in captured_prompts[0]


class TestPairToChatml:
    def test_output_has_three_messages(self) -> None:
        pair = RagPair(
            context="Context about RAG.",
            question="What is RAG?",
            answer="RAG is retrieval-augmented generation.",
        )
        result = pair_to_chatml_rag(pair)

        assert "messages" in result
        assert len(result["messages"]) == 3

    def test_roles_are_correct(self) -> None:
        pair = RagPair(
            context="Context.", question="Question?", answer="Answer."
        )
        result = pair_to_chatml_rag(pair)
        roles = [m["role"] for m in result["messages"]]
        assert roles == ["system", "user", "assistant"]

    def test_context_in_user_message(self) -> None:
        pair = RagPair(
            context="unique_context_xyz",
            question="Question?",
            answer="Answer here.",
        )
        result = pair_to_chatml_rag(pair)
        user_content = result["messages"][1]["content"]
        assert "unique_context_xyz" in user_content

    def test_answer_in_assistant_message(self) -> None:
        pair = RagPair(
            context="Context.", question="Question?", answer="unique_answer_xyz"
        )
        result = pair_to_chatml_rag(pair)
        assert result["messages"][2]["content"] == "unique_answer_xyz"
```

- [ ] **Step 2: Rodar testes para confirmar falha**

```bash
cd fase4-finetuning && uv run pytest tests/test_generate_dataset.py -v
```

Expected: `ImportError: cannot import name 'RagPair' from 'src.generate_dataset'`

- [ ] **Step 3: Implementar RAG pair generation em `src/generate_dataset.py`**

```python
"""Geração de dataset de fine-tuning para llama3.2.

Produz dois tipos de exemplos em formato ChatML:
- RAG pairs: (context, pergunta, resposta ideal) gerados dos chunks do ChromaDB
- ReAct cycles: ciclos Thought→Action→Observation→Final Answer bem-formados

Uso:
    uv run generate-dataset --rag-output data/raw/rag_pairs.jsonl \\
                            --react-output data/raw/react_cycles.jsonl \\
                            --processed-output data/processed/train.jsonl
"""

import json
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Caminho para o ChromaDB da Fase 1
FASE1_ROOT = Path(__file__).resolve().parent.parent.parent / "fase1-rag"
CHROMA_PATH = str(FASE1_ROOT / "chroma_db")
COLLECTION_NAME = "papers"
EMBEDDING_MODEL = "nomic-embed-text"
GENERATION_MODEL = "llama3.2"

# Parâmetros de filtragem
MIN_ANSWER_LEN = 50
MIN_QUESTION_LEN = 10


@dataclass
class RagPair:
    """Par (context, pergunta, resposta) para fine-tuning RAG."""

    context: str
    question: str
    answer: str


def filter_rag_pair(pair: RagPair) -> bool:
    """Verifica se um par RAG tem qualidade mínima aceitável.

    Args:
        pair: par a ser avaliado.

    Returns:
        True se o par passou nos filtros de qualidade.
    """
    return len(pair.question) >= MIN_QUESTION_LEN and len(pair.answer) >= MIN_ANSWER_LEN


def generate_rag_pair(
    chunk: str,
    llm_fn: Callable[[str], str],
) -> RagPair | None:
    """Gera um par (pergunta, resposta) a partir de um chunk de texto.

    Chama llm_fn com um prompt pedindo uma pergunta e resposta baseadas
    exclusivamente no chunk. Parseia o formato QUESTION:/ANSWER:.

    Args:
        chunk: texto do chunk recuperado do ChromaDB.
        llm_fn: função que recebe um prompt e retorna texto do LLM.

    Returns:
        RagPair se o parse e filtros passaram, None caso contrário.
    """
    prompt = (
        "Based on the following text excerpt, generate:\n"
        "1. A specific question answerable ONLY from this text\n"
        "2. A concise, accurate answer using ONLY this text\n\n"
        f"Text: {chunk}\n\n"
        "Respond in this exact format:\n"
        "QUESTION: <your question>\n"
        "ANSWER: <your answer>"
    )

    response = llm_fn(prompt)

    if "QUESTION:" not in response or "ANSWER:" not in response:
        return None

    question = ""
    answer = ""
    for line in response.strip().splitlines():
        if line.startswith("QUESTION:"):
            question = line[len("QUESTION:"):].strip()
        elif line.startswith("ANSWER:"):
            answer = line[len("ANSWER:"):].strip()

    pair = RagPair(context=chunk, question=question, answer=answer)
    return pair if filter_rag_pair(pair) else None


def pair_to_chatml_rag(pair: RagPair) -> dict:
    """Converte um RagPair para o formato ChatML multi-turn.

    Args:
        pair: par RAG com context, question e answer.

    Returns:
        Dict com chave 'messages' no formato ChatML.
    """
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful research assistant. "
                    "Answer questions based only on the provided context."
                ),
            },
            {
                "role": "user",
                "content": f"Context: {pair.context}\n\nQuestion: {pair.question}",
            },
            {"role": "assistant", "content": pair.answer},
        ]
    }


def format_observation(search_results: list[dict]) -> str:
    """Formata resultados do search_papers como string de observação.

    Args:
        search_results: lista retornada por search_papers (dicts com 'content').

    Returns:
        Texto formatado para uso como Observation no ciclo ReAct.
    """
    if not search_results:
        return "Nenhum resultado encontrado."

    parts = []
    for i, result in enumerate(search_results[:3], 1):
        content = result.get("content", "")
        parts.append(f"[{i}] {content[:300]}")
    return "\n\n".join(parts)
```

- [ ] **Step 4: Rodar testes**

```bash
cd fase4-finetuning && uv run pytest tests/test_generate_dataset.py -v
```

Expected: todos os testes passando.

- [ ] **Step 5: Lint**

```bash
cd fase4-finetuning && uv run ruff check . && uv run ruff format .
```

- [ ] **Step 6: Commit**

```bash
git add fase4-finetuning/src/generate_dataset.py fase4-finetuning/tests/test_generate_dataset.py
git commit -m "feat: generate_dataset — RAG pair generation com filtros de qualidade"
```

---

## Task 3: generate_dataset.py — ReAct cycles (TDD)

**Files:**
- Modify: `fase4-finetuning/src/generate_dataset.py` (adicionar funções ReAct)
- Modify: `fase4-finetuning/tests/test_generate_dataset.py` (adicionar testes ReAct)

- [ ] **Step 1: Adicionar testes de ReAct ao arquivo de testes**

Adicionar ao final de `fase4-finetuning/tests/test_generate_dataset.py`:

```python
from src.generate_dataset import format_observation, generate_react_cycle, pair_to_chatml_react


class TestFormatObservation:
    def test_formats_top_3_results(self) -> None:
        results = [
            {"content": "Result one about transformers."},
            {"content": "Result two about attention."},
            {"content": "Result three about BERT."},
            {"content": "Result four should be excluded."},
        ]
        observation = format_observation(results)
        assert "[1]" in observation
        assert "[2]" in observation
        assert "[3]" in observation
        assert "[4]" not in observation

    def test_empty_results_returns_fallback(self) -> None:
        observation = format_observation([])
        assert "Nenhum resultado encontrado" in observation

    def test_content_truncated_to_300_chars(self) -> None:
        long_content = "x" * 500
        results = [{"content": long_content}]
        observation = format_observation(results)
        # 300 chars + prefix "[1] "
        assert len(observation) <= 310


class TestGenerateReactCycle:
    def test_cycle_has_correct_chatml_structure(self) -> None:
        def search_fn(query: str) -> list[dict]:
            return [{"content": "Relevant paper content about the topic."}]

        def generate_fn(question: str, context: str) -> str:
            return "The answer based on the retrieved context."

        result = generate_react_cycle(
            question="What is attention mechanism?",
            search_fn=search_fn,
            generate_fn=generate_fn,
            system_prompt="You are a ReAct agent.",
        )

        assert "messages" in result
        assert len(result["messages"]) == 3
        roles = [m["role"] for m in result["messages"]]
        assert roles == ["system", "user", "assistant"]

    def test_assistant_message_has_react_keywords(self) -> None:
        def search_fn(query: str) -> list[dict]:
            return [{"content": "Paper content."}]

        def generate_fn(question: str, context: str) -> str:
            return "Final answer here."

        result = generate_react_cycle(
            question="How does RAG work?",
            search_fn=search_fn,
            generate_fn=generate_fn,
            system_prompt="System prompt.",
        )
        assistant_content = result["messages"][2]["content"]

        assert "Thought:" in assistant_content
        assert "Action: search_papers" in assistant_content
        assert "Action Input:" in assistant_content
        assert "Observation:" in assistant_content
        assert "Final Answer:" in assistant_content

    def test_search_fn_called_with_question(self) -> None:
        captured_queries = []

        def search_fn(query: str) -> list[dict]:
            captured_queries.append(query)
            return [{"content": "Some content."}]

        def generate_fn(question: str, context: str) -> str:
            return "Answer."

        generate_react_cycle(
            question="unique_question_marker",
            search_fn=search_fn,
            generate_fn=generate_fn,
            system_prompt="Prompt.",
        )

        assert len(captured_queries) == 1
        assert "unique_question_marker" in captured_queries[0]

    def test_generate_fn_called_with_observation(self) -> None:
        captured_contexts = []

        def search_fn(query: str) -> list[dict]:
            return [{"content": "unique_content_xyz"}]

        def generate_fn(question: str, context: str) -> str:
            captured_contexts.append(context)
            return "Answer."

        generate_react_cycle(
            question="Question?",
            search_fn=search_fn,
            generate_fn=generate_fn,
            system_prompt="Prompt.",
        )

        assert len(captured_contexts) == 1
        assert "unique_content_xyz" in captured_contexts[0]

    def test_system_prompt_in_first_message(self) -> None:
        def search_fn(query: str) -> list[dict]:
            return []

        def generate_fn(question: str, context: str) -> str:
            return "Answer."

        result = generate_react_cycle(
            question="Q?",
            search_fn=search_fn,
            generate_fn=generate_fn,
            system_prompt="unique_system_prompt_xyz",
        )

        assert result["messages"][0]["content"] == "unique_system_prompt_xyz"
```

- [ ] **Step 2: Rodar para confirmar falha**

```bash
cd fase4-finetuning && uv run pytest tests/test_generate_dataset.py::TestGenerateReactCycle -v
```

Expected: `ImportError: cannot import name 'generate_react_cycle'`

- [ ] **Step 3: Adicionar funções ReAct ao `src/generate_dataset.py`**

Adicionar após `format_observation`:

```python
def generate_react_cycle(
    question: str,
    search_fn: Callable[[str], list[dict]],
    generate_fn: Callable[[str, str], str],
    system_prompt: str,
) -> dict:
    """Gera um ciclo ReAct completo e bem-formado para fine-tuning.

    Em vez de capturar saídas do modelo atual (que tem formato inconsistente),
    constrói o ciclo programaticamente usando resultados reais do search_papers
    e respostas geradas pelo LLM. Isso garante exemplos com formato perfeito.

    Args:
        question: pergunta que o agente deve responder.
        search_fn: função de busca semântica (retorna list[dict] com 'content').
        generate_fn: função que gera resposta dado (question, context).
        system_prompt: prompt de sistema para o papel 'system' do ChatML.

    Returns:
        Dict no formato ChatML com ciclo ReAct completo.
    """
    # Executar busca real para obter observação autêntica
    search_results = search_fn(question)
    observation = format_observation(search_results)

    # Gerar resposta final baseada no contexto recuperado
    final_answer = generate_fn(question, observation)

    # Construir ciclo com formato perfeito
    assistant_content = (
        "Thought: I need to search the research papers to find relevant "
        f"information about this question.\n"
        f"Action: search_papers\n"
        f"Action Input: {question}\n"
        f"Observation: {observation}\n"
        "Thought: Based on the search results, I have enough information "
        "to provide an accurate answer.\n"
        f"Final Answer: {final_answer}"
    )

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def pair_to_chatml_react(cycle: dict) -> dict:
    """Passa o ciclo direto — já está em formato ChatML.

    Existe para simetria com pair_to_chatml_rag e para facilitar testes.

    Args:
        cycle: dict já no formato ChatML retornado por generate_react_cycle.

    Returns:
        O mesmo dict (identidade).
    """
    return cycle
```

- [ ] **Step 4: Rodar todos os testes**

```bash
cd fase4-finetuning && uv run pytest tests/test_generate_dataset.py -v
```

Expected: todos os testes passando.

- [ ] **Step 5: Lint**

```bash
cd fase4-finetuning && uv run ruff check . && uv run ruff format .
```

- [ ] **Step 6: Commit**

```bash
git add fase4-finetuning/src/generate_dataset.py fase4-finetuning/tests/test_generate_dataset.py
git commit -m "feat: generate_dataset — geração de ciclos ReAct sintéticos bem-formados"
```

---

## Task 4: generate_dataset.py — preprocessing + CLI (TDD)

**Files:**
- Modify: `fase4-finetuning/src/generate_dataset.py` (preprocess + main)
- Modify: `fase4-finetuning/tests/test_generate_dataset.py` (testes preprocess)

- [ ] **Step 1: Adicionar testes de preprocessing**

Adicionar ao final de `fase4-finetuning/tests/test_generate_dataset.py`:

```python
import json
import tempfile
from pathlib import Path

from src.generate_dataset import preprocess_dataset


class TestPreprocessDataset:
    def _write_jsonl(self, path: Path, items: list[dict]) -> None:
        with open(path, "w") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")

    def test_output_files_created(self) -> None:
        rag_item = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}
        react_item = {"messages": [{"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"}]}

        with tempfile.TemporaryDirectory() as tmpdir:
            rag_path = Path(tmpdir) / "rag.jsonl"
            react_path = Path(tmpdir) / "react.jsonl"
            output_path = Path(tmpdir) / "processed" / "train.jsonl"

            self._write_jsonl(rag_path, [rag_item] * 10)
            self._write_jsonl(react_path, [react_item] * 10)

            result = preprocess_dataset(str(rag_path), str(react_path), str(output_path))

            assert output_path.exists()
            assert output_path.parent / "val.jsonl"
            assert "train" in result
            assert "val" in result

    def test_total_count_preserved(self) -> None:
        rag_item = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}
        react_item = {"messages": [{"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"}]}

        with tempfile.TemporaryDirectory() as tmpdir:
            rag_path = Path(tmpdir) / "rag.jsonl"
            react_path = Path(tmpdir) / "react.jsonl"
            output_path = Path(tmpdir) / "processed" / "train.jsonl"

            self._write_jsonl(rag_path, [rag_item] * 100)
            self._write_jsonl(react_path, [react_item] * 100)

            result = preprocess_dataset(str(rag_path), str(react_path), str(output_path))

            assert result["train"] + result["val"] == 200

    def test_val_ratio_respected(self) -> None:
        rag_item = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}
        react_item = {"messages": [{"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"}]}

        with tempfile.TemporaryDirectory() as tmpdir:
            rag_path = Path(tmpdir) / "rag.jsonl"
            react_path = Path(tmpdir) / "react.jsonl"
            output_path = Path(tmpdir) / "processed" / "train.jsonl"

            self._write_jsonl(rag_path, [rag_item] * 50)
            self._write_jsonl(react_path, [react_item] * 50)

            result = preprocess_dataset(
                str(rag_path), str(react_path), str(output_path), val_ratio=0.1
            )

            assert result["val"] == 10  # 10% de 100

    def test_reproducible_with_same_seed(self) -> None:
        rag_item = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}
        react_item = {"messages": [{"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"}]}

        with tempfile.TemporaryDirectory() as tmpdir:
            rag_path = Path(tmpdir) / "rag.jsonl"
            react_path = Path(tmpdir) / "react.jsonl"

            self._write_jsonl(rag_path, [{"messages": [{"role": "user", "content": str(i)}]} for i in range(30)])
            self._write_jsonl(react_path, [{"messages": [{"role": "user", "content": str(i + 100)}]} for i in range(30)])

            output1 = Path(tmpdir) / "out1" / "train.jsonl"
            output2 = Path(tmpdir) / "out2" / "train.jsonl"

            preprocess_dataset(str(rag_path), str(react_path), str(output1), seed=42)
            preprocess_dataset(str(rag_path), str(react_path), str(output2), seed=42)

            assert output1.read_text() == output2.read_text()
```

- [ ] **Step 2: Rodar para confirmar falha**

```bash
cd fase4-finetuning && uv run pytest tests/test_generate_dataset.py::TestPreprocessDataset -v
```

Expected: `ImportError: cannot import name 'preprocess_dataset'`

- [ ] **Step 3: Adicionar `preprocess_dataset` e `main` ao `src/generate_dataset.py`**

Adicionar ao final do arquivo:

```python
def preprocess_dataset(
    rag_path: str,
    react_path: str,
    output_path: str,
    rag_ratio: float = 0.6,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, int]:
    """Combina, balanceia e divide os datasets RAG e ReAct.

    Realiza shuffle determinístico e split treino/validação.

    Args:
        rag_path: caminho para rag_pairs.jsonl.
        react_path: caminho para react_cycles.jsonl.
        output_path: caminho para train.jsonl de saída.
        rag_ratio: fração do total a ser preenchida com exemplos RAG.
        val_ratio: fração a ser separada para validação.
        seed: semente para reprodutibilidade.

    Returns:
        Dict com contagens {'train': N, 'val': M}.
    """
    import random

    def _read_jsonl(path: str) -> list[dict]:
        return [
            json.loads(line)
            for line in Path(path).read_text(encoding="utf-8").strip().splitlines()
            if line.strip()
        ]

    rag_examples = _read_jsonl(rag_path)
    react_examples = _read_jsonl(react_path)

    # Balancear proporções
    total = len(rag_examples) + len(react_examples)
    target_rag = int(total * rag_ratio)
    target_react = total - target_rag

    random.seed(seed)
    if len(rag_examples) > target_rag:
        rag_examples = random.sample(rag_examples, target_rag)
    if len(react_examples) > target_react:
        react_examples = random.sample(react_examples, target_react)

    all_examples = rag_examples + react_examples
    random.shuffle(all_examples)

    # Split treino/validação
    val_size = int(len(all_examples) * val_ratio)
    val_examples = all_examples[:val_size]
    train_examples = all_examples[val_size:]

    # Salvar
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    val_path = out_path.parent / "val.jsonl"

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in train_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for ex in val_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    return {"train": len(train_examples), "val": len(val_examples)}


def _make_default_llm_fn(model: str = GENERATION_MODEL) -> Callable[[str], str]:
    """Cria uma função LLM usando ollama.chat."""
    import ollama

    def llm_fn(prompt: str) -> str:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.message.content

    return llm_fn


def _make_default_search_fn() -> Callable[[str], list[dict]]:
    """Cria a função de busca usando o search_papers da Fase 3."""
    fase3_src = Path(__file__).resolve().parent.parent.parent / "fase3-agents" / "src"
    if str(fase3_src) not in sys.path:
        sys.path.insert(0, str(fase3_src.parent))

    from src.tools import search_papers  # noqa: PLC0415

    return search_papers


def _make_default_generate_fn(model: str = GENERATION_MODEL) -> Callable[[str, str], str]:
    """Cria a função generate para ciclos ReAct."""
    import ollama

    def generate_fn(question: str, context: str) -> str:
        system = (
            "Answer the question based only on the provided context. "
            "Be concise and factual.\n\nContext:\n" + context
        )
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": question},
            ],
        )
        return response.message.content

    return generate_fn


def generate_rag_dataset(
    output_path: str,
    chroma_path: str = CHROMA_PATH,
    collection_name: str = COLLECTION_NAME,
    max_pairs: int = 800,
    sample_every: int = 5,
    model: str = GENERATION_MODEL,
    llm_fn: Callable[[str], str] | None = None,
) -> int:
    """Gera o dataset de pares RAG a partir do ChromaDB da Fase 1.

    Args:
        output_path: caminho para salvar rag_pairs.jsonl.
        chroma_path: caminho para o ChromaDB.
        collection_name: nome da collection no ChromaDB.
        max_pairs: número máximo de pares a gerar.
        sample_every: amostrar 1 em cada N chunks (para reduzir tempo).
        model: nome do modelo Ollama.
        llm_fn: função LLM para injeção em testes (usa Ollama se None).

    Returns:
        Número de pares gerados e salvos.
    """
    import chromadb

    if llm_fn is None:
        llm_fn = _make_default_llm_fn(model)

    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_collection(name=collection_name)
    all_docs = collection.get(include=["documents"])

    documents = all_docs.get("documents", [])
    sampled = documents[::sample_every][:max_pairs]

    pairs = []
    for i, chunk in enumerate(sampled):
        logger.info("RAG pair %d/%d", i + 1, len(sampled))
        pair = generate_rag_pair(chunk, llm_fn)
        if pair:
            pairs.append(pair_to_chatml_rag(pair))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    return len(pairs)


def generate_react_dataset(
    output_path: str,
    questions: list[str] | None = None,
    search_fn: Callable[[str], list[dict]] | None = None,
    generate_fn: Callable[[str, str], str] | None = None,
    model: str = GENERATION_MODEL,
) -> int:
    """Gera o dataset de ciclos ReAct usando busca real + LLM.

    Args:
        output_path: caminho para salvar react_cycles.jsonl.
        questions: lista de perguntas seed (usa default se None).
        search_fn: função de busca (usa search_papers da Fase 3 se None).
        generate_fn: função de geração (usa Ollama se None).
        model: nome do modelo Ollama.

    Returns:
        Número de ciclos gerados e salvos.
    """
    if questions is None:
        questions = _default_react_questions()

    if search_fn is None:
        search_fn = _make_default_search_fn()

    if generate_fn is None:
        generate_fn = _make_default_generate_fn(model)

    # Importar SYSTEM_PROMPT da Fase 3
    fase3_src = Path(__file__).resolve().parent.parent.parent / "fase3-agents"
    if str(fase3_src) not in sys.path:
        sys.path.insert(0, str(fase3_src))

    from src.prompt import SYSTEM_PROMPT  # noqa: PLC0415

    cycles = []
    for i, question in enumerate(questions):
        logger.info("ReAct cycle %d/%d: %s", i + 1, len(questions), question[:50])
        cycle = generate_react_cycle(question, search_fn, generate_fn, SYSTEM_PROMPT)
        cycles.append(cycle)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for c in cycles:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    return len(cycles)


def _default_react_questions() -> list[str]:
    """Carrega perguntas do dataset da Fase 2 como seed para ciclos ReAct."""
    fase2_dataset = (
        Path(__file__).resolve().parent.parent.parent
        / "fase2-evals"
        / "data"
        / "eval_dataset.json"
    )
    if fase2_dataset.exists():
        import json as _json

        entries = _json.loads(fase2_dataset.read_text(encoding="utf-8"))
        return [e["question"] for e in entries]

    # Fallback se o dataset não estiver disponível
    return [
        "What are the main advantages of retrieval-augmented generation?",
        "How does the attention mechanism work in transformer models?",
        "What metrics are commonly used to evaluate language models?",
        "How does LoRA reduce the number of trainable parameters?",
        "What is the difference between RAG and fine-tuning?",
    ]


def main() -> None:
    """CLI para geração do dataset completo."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Gera dataset de fine-tuning")
    parser.add_argument("--rag-output", default="data/raw/rag_pairs.jsonl")
    parser.add_argument("--react-output", default="data/raw/react_cycles.jsonl")
    parser.add_argument("--processed-output", default="data/processed/train.jsonl")
    parser.add_argument("--max-rag-pairs", type=int, default=800)
    parser.add_argument("--model", default=GENERATION_MODEL)
    args = parser.parse_args()

    logger.info("Gerando RAG pairs...")
    n_rag = generate_rag_dataset(args.rag_output, max_pairs=args.max_rag_pairs, model=args.model)
    logger.info("RAG pairs gerados: %d", n_rag)

    logger.info("Gerando ciclos ReAct...")
    n_react = generate_react_dataset(args.react_output, model=args.model)
    logger.info("Ciclos ReAct gerados: %d", n_react)

    logger.info("Preprocessando dataset...")
    counts = preprocess_dataset(args.rag_output, args.react_output, args.processed_output)
    logger.info("Dataset final: %d treino, %d validação", counts["train"], counts["val"])
```

- [ ] **Step 4: Rodar todos os testes**

```bash
cd fase4-finetuning && uv run pytest tests/test_generate_dataset.py -v
```

Expected: todos os testes passando.

- [ ] **Step 5: Lint**

```bash
cd fase4-finetuning && uv run ruff check . && uv run ruff format .
```

- [ ] **Step 6: Commit**

```bash
git add fase4-finetuning/src/generate_dataset.py fase4-finetuning/tests/test_generate_dataset.py
git commit -m "feat: generate_dataset — preprocessing, CLI e funções de integração completas"
```

---

## Task 5: train.py (TDD)

**Files:**
- Create: `fase4-finetuning/src/train.py`
- Create: `fase4-finetuning/tests/test_train.py`

- [ ] **Step 1: Escrever testes para train.py**

Criar `fase4-finetuning/tests/test_train.py`:

```python
"""Testes para o módulo de treinamento QLoRA."""

import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from src.train import TrainConfig, load_and_format_dataset


class TestTrainConfig:
    def test_default_values(self) -> None:
        config = TrainConfig()
        assert config.model_name == "unsloth/llama-3.2-3b-instruct"
        assert config.r == 16
        assert config.lora_alpha == 16
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
        mock_tokenizer.apply_chat_template.return_value = "<|system|>System<|user|>Q<|assistant|>A"

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
        example = {"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]}

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "text"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(example) + "\n")
            f.write("\n")  # linha vazia
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
        import tempfile

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "text"
        mock_flm.from_pretrained.return_value = (mock_model, mock_tokenizer)
        mock_flm.get_peft_model.return_value = mock_model

        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "train.jsonl"
            example = {"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]}
            dataset_path.write_text(json.dumps(example) + "\n")

            output_dir = Path(tmpdir) / "adapter"
            from src.train import TrainConfig, train

            config = TrainConfig(
                dataset_path=str(dataset_path),
                output_dir=str(output_dir),
            )
            result = train(config)

        mock_trainer.train.assert_called_once()
        mock_model.save_pretrained.assert_called_once()
        mock_tokenizer.save_pretrained.assert_called_once()

    @patch("src.train.SFTTrainer")
    @patch("src.train.TrainingArguments")
    @patch("src.train.FastLanguageModel")
    def test_train_returns_output_path(
        self, mock_flm: MagicMock, mock_args: MagicMock, mock_trainer_cls: MagicMock
    ) -> None:
        import tempfile

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "text"
        mock_flm.from_pretrained.return_value = (mock_model, mock_tokenizer)
        mock_flm.get_peft_model.return_value = mock_model
        mock_trainer_cls.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "train.jsonl"
            example = {"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]}
            dataset_path.write_text(json.dumps(example) + "\n")

            output_dir = Path(tmpdir) / "adapter"
            from src.train import TrainConfig, train

            config = TrainConfig(
                dataset_path=str(dataset_path),
                output_dir=str(output_dir),
            )
            result = train(config)

        assert result == output_dir
```

- [ ] **Step 2: Rodar para confirmar falha**

```bash
cd fase4-finetuning && uv run pytest tests/test_train.py -v
```

Expected: `ImportError: cannot import name 'TrainConfig' from 'src.train'`

- [ ] **Step 3: Criar `src/train.py`**

```python
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
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Importações no nível do módulo para permitir @patch("src.train.X") nos testes.
# try/except porque requerem GPU + instalação especial do Unsloth.
try:
    import torch  # noqa: F401
    from trl import SFTTrainer  # noqa: F401
    from transformers import TrainingArguments  # noqa: F401
    from unsloth import FastLanguageModel  # noqa: F401
except ImportError:
    # Instalar com: uv run pip install "unsloth[cu124-torch250]" trl peft
    torch = None  # type: ignore[assignment]
    FastLanguageModel = None  # type: ignore[assignment]
    SFTTrainer = None  # type: ignore[assignment]
    TrainingArguments = None  # type: ignore[assignment]


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
    lora_alpha: int = 16
    lora_dropout: float = 0.0

    # Treinamento
    batch_size: int = 4
    grad_accumulation: int = 4
    learning_rate: float = 2e-4
    num_epochs: int = 3
    warmup_steps: int = 5
    seed: int = 42


def load_and_format_dataset(path: str, tokenizer) -> "datasets.Dataset":
    """Carrega JSONL no formato ChatML e aplica o template de chat.

    Converte cada exemplo de {'messages': [...]} para {'text': '<chat formatted>'}.

    Args:
        path: caminho para arquivo .jsonl com exemplos em formato ChatML.
        tokenizer: tokenizer com método apply_chat_template.

    Returns:
        HuggingFace Dataset com coluna 'text'.
    """
    from datasets import Dataset

    examples = []
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
            examples.append({"text": text})

    return Dataset.from_list(examples)


def train(config: TrainConfig) -> Path:
    """Executa o treinamento QLoRA e salva o adapter LoRA.

    Args:
        config: TrainConfig com todos os hiperparâmetros.

    Returns:
        Path para o diretório onde o adapter foi salvo.
    """
    if FastLanguageModel is None:
        raise ImportError(
            "Unsloth não instalado. Execute: uv run pip install 'unsloth[cu124-torch250]' trl peft"
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
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config.seed,
    )

    logger.info("Carregando dataset: %s", config.dataset_path)
    train_dataset = load_and_format_dataset(config.dataset_path, tokenizer)
    logger.info("Dataset carregado: %d exemplos", len(train_dataset))

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        dataset_num_proc=2,
        args=TrainingArguments(
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.grad_accumulation,
            warmup_steps=config.warmup_steps,
            num_train_epochs=config.num_epochs,
            learning_rate=config.learning_rate,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
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
```

- [ ] **Step 4: Rodar todos os testes**

```bash
cd fase4-finetuning && uv run pytest tests/test_train.py -v
```

Expected: todos os testes passando (mocks cobrem Unsloth/TRL).

- [ ] **Step 5: Lint**

```bash
cd fase4-finetuning && uv run ruff check . && uv run ruff format .
```

- [ ] **Step 6: Commit**

```bash
git add fase4-finetuning/src/train.py fase4-finetuning/tests/test_train.py
git commit -m "feat: train.py — TrainConfig + loop QLoRA com Unsloth/TRL"
```

---

## Task 6: export.py (TDD)

**Files:**
- Create: `fase4-finetuning/src/export.py`
- Create: `fase4-finetuning/tests/test_export.py`

- [ ] **Step 1: Escrever testes para export.py**

Criar `fase4-finetuning/tests/test_export.py`:

```python
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
            # Criar um arquivo .gguf falso
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
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
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
```

- [ ] **Step 2: Rodar para confirmar falha**

```bash
cd fase4-finetuning && uv run pytest tests/test_export.py -v
```

Expected: `ImportError: cannot import name 'create_modelfile' from 'src.export'`

- [ ] **Step 3: Criar `src/export.py`**

```python
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
    from unsloth import FastLanguageModel

    logger.info("Carregando adapter: %s", adapter_path)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    output_path = Path(output_gguf_path)
    output_path.mkdir(parents=True, exist_ok=True)

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
        f"FROM ./{gguf_file.name}\n"
        "PARAMETER temperature 0.7\n"
        "PARAMETER num_ctx 4096\n"
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
```

- [ ] **Step 4: Rodar todos os testes**

```bash
cd fase4-finetuning && uv run pytest tests/test_export.py -v
```

Expected: todos os testes passando.

- [ ] **Step 5: Lint**

```bash
cd fase4-finetuning && uv run ruff check . && uv run ruff format .
```

- [ ] **Step 6: Commit**

```bash
git add fase4-finetuning/src/export.py fase4-finetuning/tests/test_export.py
git commit -m "feat: export.py — adapter → GGUF → ollama create"
```

---

## Task 7: evaluate.py + integration tests

**Files:**
- Create: `fase4-finetuning/src/evaluate.py`
- Create: `fase4-finetuning/tests/test_evaluate.py` (para os helpers)

- [ ] **Step 1: Escrever testes para os helpers de compare**

Criar `fase4-finetuning/tests/test_evaluate.py`:

```python
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
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
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
    def test_context_in_system_message(self, mock_ollama: MagicMock) -> None:
        mock_ollama.chat.return_value = MagicMock(message=MagicMock(content="A."))

        generate_fn = make_generate_fn("model")
        generate_fn("Q?", "unique_context_xyz")

        call_args = mock_ollama.chat.call_args
        messages = call_args.kwargs["messages"]
        system_content = messages[0]["content"]
        assert "unique_context_xyz" in system_content
```

- [ ] **Step 2: Rodar para confirmar falha**

```bash
cd fase4-finetuning && uv run pytest tests/test_evaluate.py -v
```

Expected: `ImportError: cannot import name '_compare_results' from 'src.evaluate'`

- [ ] **Step 3: Criar `src/evaluate.py`**

```python
"""Avaliação comparativa: baseline (Fase 2) vs. modelo fine-tuned.

Reutiliza o sistema de evals da Fase 2 com o modelo parametrizado.

Uso:
    uv run evaluate-model --model llama3.2-finetuned \\
                          --output data/results
"""

import argparse
import json
import logging
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import ollama

logger = logging.getLogger(__name__)

FASE2_ROOT = Path(__file__).resolve().parent.parent.parent / "fase2-evals"
DEFAULT_BASELINE_PATH = str(FASE2_ROOT / "data" / "results")
DEFAULT_DATASET_PATH = str(FASE2_ROOT / "data" / "eval_dataset.json")
EMBEDDING_MODEL = "nomic-embed-text"
FASE1_CHROMA_PATH = str(
    Path(__file__).resolve().parent.parent.parent / "fase1-rag" / "chroma_db"
)


def make_generate_fn(model_name: str) -> Callable[[str, str], str]:
    """Cria uma função generate parametrizada pelo nome do modelo Ollama.

    Permite trocar o modelo de geração sem alterar o evaluator da Fase 2.

    Args:
        model_name: nome do modelo no Ollama (ex: 'llama3.2-finetuned').

    Returns:
        Função compatível com a assinatura esperada pelo evaluate_dataset da Fase 2.
    """
    def generate_fn(question: str, context: str) -> str:
        system = (
            "You are an assistant that answers questions exclusively based on the "
            "provided context. If the answer is not in the context, say so clearly.\n\n"
            f"Context:\n{context}"
        )
        response = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": question},
            ],
        )
        return response.message.content

    return generate_fn


def _load_fase2_functions():
    """Importa evaluate_dataset, load_dataset e save_report da Fase 2."""
    fase2_src = str(FASE2_ROOT)
    if fase2_src not in sys.path:
        sys.path.insert(0, fase2_src)

    from src.dataset import load_dataset  # noqa: PLC0415
    from src.evaluator import evaluate_dataset  # noqa: PLC0415
    from src.report import save_report  # noqa: PLC0415

    return evaluate_dataset, load_dataset, save_report


def _make_retrieve_fn() -> Callable:
    """Cria a função de retrieval usando ChromaDB da Fase 1."""
    import chromadb

    def retrieve_fn(
        query: str,
        top_k: int = 5,
        collection_name: str = "papers",
        chroma_path: str = FASE1_CHROMA_PATH,
    ) -> list[dict]:
        client = chromadb.PersistentClient(path=chroma_path)
        collection = client.get_collection(name=collection_name)

        n_results = min(top_k, collection.count())
        if n_results == 0:
            return []

        embedding_response = ollama.embed(model=EMBEDDING_MODEL, input=query)
        embedding = embedding_response.embeddings[0]

        results = collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        return [
            {
                "id": chunk_id,
                "text": doc,
                "source": meta.get("source", ""),
                "page": meta.get("page", 0),
                "score": round(1 - dist, 4),
            }
            for chunk_id, doc, meta, dist in zip(ids, docs, metas, distances, strict=True)
        ]

    return retrieve_fn


def _load_latest_baseline(baseline_path: str) -> dict:
    """Carrega o arquivo de resultado mais recente da Fase 2.

    Args:
        baseline_path: diretório com arquivos eval_fase1_*.json.

    Returns:
        Dict com resultados do baseline.

    Raises:
        FileNotFoundError: se nenhum baseline for encontrado.
    """
    baseline_dir = Path(baseline_path)
    json_files = sorted(baseline_dir.glob("eval_fase1_*.json"))

    if not json_files:
        raise FileNotFoundError(
            f"Nenhum baseline encontrado em: {baseline_path}\n"
            "Execute a Fase 2 primeiro: cd ../fase2-evals && uv run eval-run"
        )

    return json.loads(json_files[-1].read_text(encoding="utf-8"))


def _compare_results(
    current: dict,
    baseline: dict,
    model_name: str,
) -> dict:
    """Gera relatório comparativo entre baseline e resultado atual.

    Args:
        current: EvalReport da Fase 2 com avg_faithfulness e avg_answer_relevance.
        baseline: EvalReport da Fase 2 do baseline (Fase 1 original).
        model_name: nome do modelo avaliado.

    Returns:
        Dict com métricas comparativas, deltas e flags de meta atingida.
    """
    def _metric(key: str, target: float) -> dict:
        baseline_val = baseline.get(key, 0.0)
        current_val = current.get(key, 0.0)
        return {
            "baseline": round(baseline_val, 4),
            "finetuned": round(current_val, 4),
            "delta": round(current_val - baseline_val, 4),
            "target": target,
            "achieved": current_val > target,
        }

    return {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "faithfulness": _metric("avg_faithfulness", target=0.70),
            "answer_relevance": _metric("avg_answer_relevance", target=0.70),
        },
        "full_result": current,
        "full_baseline": baseline,
    }


def run_evaluation(
    model_name: str = "llama3.2-finetuned",
    baseline_path: str = DEFAULT_BASELINE_PATH,
    dataset_path: str = DEFAULT_DATASET_PATH,
    output_dir: str = "data/results",
) -> dict:
    """Executa a avaliação da Fase 2 com o modelo fine-tuned e compara com baseline.

    Args:
        model_name: nome do modelo no Ollama a avaliar.
        baseline_path: diretório com resultados do baseline da Fase 2.
        dataset_path: caminho para eval_dataset.json da Fase 2.
        output_dir: diretório para salvar o relatório comparativo.

    Returns:
        Dict com comparação baseline vs. fine-tuned.
    """
    evaluate_dataset, load_dataset, save_report = _load_fase2_functions()

    logger.info("Carregando dataset de avaliação: %s", dataset_path)
    entries = load_dataset(Path(dataset_path))
    logger.info("Dataset: %d perguntas", len(entries))

    logger.info("Executando avaliação com modelo: %s", model_name)
    report = evaluate_dataset(
        entries,
        retrieve_fn=_make_retrieve_fn(),
        generate_fn=make_generate_fn(model_name),
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = f"fase4_{timestamp}"
    result_path = save_report(report, label=label)
    logger.info("Resultado salvo em: %s", result_path)

    logger.info("Carregando baseline para comparação...")
    baseline = _load_latest_baseline(baseline_path)

    comparison = _compare_results(report, baseline, model_name)

    # Salvar comparação
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    comparison_path = Path(output_dir) / f"comparison_{timestamp}.json"
    comparison_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False))
    logger.info("Comparação salva em: %s", comparison_path)

    # Imprimir sumário
    _print_summary(comparison)

    return comparison


def _print_summary(comparison: dict) -> None:
    """Imprime sumário dos resultados no terminal."""
    metrics = comparison["metrics"]

    print("\n" + "=" * 50)
    print(f"AVALIAÇÃO: {comparison['model']}")
    print("=" * 50)
    for name, data in metrics.items():
        status = "✓" if data["achieved"] else "✗"
        delta_str = f"+{data['delta']:.3f}" if data["delta"] >= 0 else f"{data['delta']:.3f}"
        print(
            f"{status} {name:20s}: {data['baseline']:.3f} → {data['finetuned']:.3f} "
            f"({delta_str}) | meta: >{data['target']:.2f}"
        )
    print("=" * 50 + "\n")


def main() -> None:
    """CLI para avaliação comparativa."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Avalia modelo fine-tuned vs. baseline")
    parser.add_argument("--model", default="llama3.2-finetuned")
    parser.add_argument("--baseline-path", default=DEFAULT_BASELINE_PATH)
    parser.add_argument("--output", default="data/results")
    args = parser.parse_args()

    run_evaluation(
        model_name=args.model,
        baseline_path=args.baseline_path,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Rodar todos os testes**

```bash
cd fase4-finetuning && uv run pytest tests/test_evaluate.py -v
```

Expected: todos os testes passando.

- [ ] **Step 5: Rodar suite completa**

```bash
cd fase4-finetuning && uv run pytest tests/ -v
```

Expected: todos os testes passando.

- [ ] **Step 6: Lint final**

```bash
cd fase4-finetuning && uv run ruff check . && uv run ruff format .
```

- [ ] **Step 7: Commit**

```bash
git add fase4-finetuning/src/evaluate.py fase4-finetuning/tests/test_evaluate.py
git commit -m "feat: evaluate.py — avaliação comparativa baseline vs. fine-tuned"
```

---

## Task 8: Final integration test + Obsidian docs

**Files:**
- Create: `fase4-finetuning/tests/test_integration.py`

- [ ] **Step 1: Criar testes de integração (marcados, não rodam no CI)**

Criar `fase4-finetuning/tests/test_integration.py`:

```python
"""Testes de integração para a Fase 4.

Requerem: Ollama rodando com llama3.2, ChromaDB da Fase 1 populado.
Executar com: pytest tests/test_integration.py -m integration -v
"""

import json
from pathlib import Path

import pytest

FASE1_CHROMA = Path(__file__).resolve().parent.parent.parent / "fase1-rag" / "chroma_db"


@pytest.mark.integration
def test_rag_pair_generation_with_real_chromadb() -> None:
    """Gera 3 pares RAG reais a partir do ChromaDB da Fase 1."""
    import chromadb

    from src.generate_dataset import generate_rag_dataset

    assert FASE1_CHROMA.exists(), f"ChromaDB não encontrado: {FASE1_CHROMA}"

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "rag_pairs.jsonl"
        n_pairs = generate_rag_dataset(
            str(output_path),
            chroma_path=str(FASE1_CHROMA),
            max_pairs=3,
            sample_every=100,
        )

    assert n_pairs >= 1
    assert output_path.exists()


@pytest.mark.integration
def test_react_cycle_generation_with_real_tools() -> None:
    """Gera 2 ciclos ReAct reais usando ChromaDB e Ollama."""
    from src.generate_dataset import generate_react_dataset

    assert FASE1_CHROMA.exists(), f"ChromaDB não encontrado: {FASE1_CHROMA}"

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "react_cycles.jsonl"
        questions = [
            "What are the main advantages of retrieval-augmented generation?",
            "How does the attention mechanism work?",
        ]
        n_cycles = generate_react_dataset(str(output_path), questions=questions)

    assert n_cycles == 2
    assert output_path.exists()

    # Verificar formato correto dos ciclos
    cycles = [json.loads(line) for line in output_path.read_text().splitlines() if line]
    for cycle in cycles:
        assistant_content = cycle["messages"][2]["content"]
        assert "Thought:" in assistant_content
        assert "Action: search_papers" in assistant_content
        assert "Final Answer:" in assistant_content
```

- [ ] **Step 2: Rodar testes unitários (não integration)**

```bash
cd fase4-finetuning && uv run pytest tests/ -v --ignore=tests/test_integration.py
```

Expected: todos passando.

- [ ] **Step 3: Criar nota no Obsidian**

Criar arquivo `/home/lbr/obsidian-vault/AI Engineering Journey/Fase 4 - Fine-tuning/01 - Fundamentos e Decisões.md`:

```markdown
# Fase 4 — Fundamentos e Decisões de Arquitetura

## Objetivo

Fine-tuning supervisionado (SFT) com QLoRA para melhorar dois problemas
identificados nas Fases 2 e 3:

- **Faithfulness**: 0.57 → >0.70 (o modelo gera afirmações não suportadas pelos chunks)
- **Answer Relevance**: 0.60 → >0.70 (respostas nem sempre endereçam a pergunta)
- **Formato ReAct**: ~70% → >90% (llama3.2 3B polui Action Input com texto extra)

## Decisões

### Por que QLoRA e não full fine-tuning?
Full fine-tuning do llama3.2 3B exige ~24GB VRAM. QLoRA treina apenas matrizes
de rank baixo (r=16) nos pesos de atenção, reduzindo para ~6GB com quantização 4-bit.
Trade-off: qualidade ligeiramente menor, mas viável em hardware consumer.

### Por que Unsloth e não TRL puro?
Unsloth reescreve kernels CUDA para atenção e gradient checkpointing, resultando
em 2x mais velocidade e 30% menos memória que TRL padrão. Para hardware local,
isso é a diferença entre treinar em 2h ou 4h.

### Por que dataset sintético e não humano?
Temos 4.110 chunks do ArXiv já indexados. Usar LLM-as-teacher (llama3.2 gerando
pares question/answer) é mais rápido e escala com o corpus existente. O risco
é "aprender os erros do teacher", mas para RAG quality a meta é comportamento
mais conservador (não alucinar), o que o teacher já faz razoavelmente.

### Por que ciclos ReAct sintéticos (template) e não capturados?
O llama3.2 3B tem formato ReAct inconsistente — capturar suas saídas significaria
treinar em exemplos com o problema que queremos corrigir. Construir ciclos
programaticamente (busca real, template de formato, LLM para conteúdo) garante
exemplos 100% bem-formados como ground truth.

### Por que export GGUF para Ollama?
Consistência com as Fases 1-3: toda a stack roda local via Ollama. O Unsloth
exporta para GGUF nativamente via `save_pretrained_gguf()`, evitando dependência
manual do llama.cpp.
```

- [ ] **Step 4: Rodar sync do Obsidian**

```bash
~/sync-obsidian.sh
```

- [ ] **Step 5: Commit final**

```bash
git add fase4-finetuning/tests/test_integration.py
git commit -m "test: testes de integração para geração de dataset da Fase 4"
```

---

## Verificação End-to-End

Após todos os tasks, verificar que o pipeline completo funciona:

```bash
# 1. Todos os testes unitários passando
cd fase4-finetuning && uv run pytest tests/ -v --ignore=tests/test_integration.py
# Expected: todos passando

# 2. Lint limpo
uv run ruff check . && uv run ruff format --check .
# Expected: sem erros

# 3. Smoke test da geração (requer Ollama + ChromaDB)
# uv run generate-dataset --max-rag-pairs 5 --rag-output /tmp/test_rag.jsonl \
#                         --react-output /tmp/test_react.jsonl \
#                         --processed-output /tmp/test_train.jsonl
# Expected: arquivos criados com exemplos ChatML válidos

# 4. Smoke test do treinamento (requer GPU + Unsloth instalado)
# uv run train-model --dataset /tmp/test_train.jsonl --output /tmp/test_adapter --epochs 1
# Expected: adapter salvo em /tmp/test_adapter/

# 5. Smoke test do export (requer Unsloth + Ollama)
# uv run export-model --adapter /tmp/test_adapter --gguf-output /tmp/test_gguf
# Expected: arquivo .gguf criado e modelo registrado no Ollama

# 6. Avaliação completa (requer Fase 2 baseline + modelo registrado)
# uv run evaluate-model --model llama3.2-finetuned
# Expected: relatório comparativo mostrando melhoria vs. baseline
```
