# Design Spec — Fase 4: Fine-tuning

**Data:** 2026-04-12  
**Status:** Aprovado  
**Autor:** Leonardo Brasil

---

## Contexto

As Fases 1-3 construíram um pipeline RAG completo, um sistema de evals quantitativo e um
agente ReAct do zero. Os resultados reais medidos na Fase 2 revelaram dois gargalos:

- **Faithfulness: 0.57** — o modelo frequentemente gera afirmações não suportadas pelos chunks
- **Answer Relevance: 0.60** — respostas nem sempre endereçam diretamente a pergunta

A Fase 3 expôs um terceiro problema: o llama3.2 (3B) segue o formato ReAct de forma
inconsistente (~70%), poluindo `Action Input` com texto extra e quebrando o parser.

O objetivo da Fase 4 é aplicar fine-tuning supervisionado (SFT) com QLoRA para atacar
ambos os problemas com o mesmo modelo já em uso no projeto.

---

## Objetivos e Métricas de Sucesso

| Métrica | Baseline (Fase 2) | Meta |
|---|---|---|
| Faithfulness médio | 0.57 | > 0.70 |
| Answer Relevance médio | 0.60 | > 0.70 |
| Formato ReAct correto (%) | ~70% | > 90% |

---

## Decisões de Design

| Dimensão | Escolha | Justificativa |
|---|---|---|
| Hardware | GPU local ≥8GB VRAM | Disponível, evita cloud |
| Modelo base | llama3.2 3B Instruct | Já em uso, comparação direta baseline vs. fine-tuned |
| Técnica | QLoRA (4-bit) | ~6GB VRAM, viável no hardware disponível |
| Framework | Unsloth + TRL SFTTrainer | 2x mais rápido que TRL puro |
| Dataset | Misto RAG + ReAct, formato ChatML | Cobre ambos os objetivos em um treinamento |
| Export | GGUF via llama.cpp → Ollama | Consistente com setup das fases anteriores |

---

## Arquitetura

```
fase4-finetuning/
├── CLAUDE.md
├── pyproject.toml
├── data/
│   ├── raw/
│   │   ├── rag_pairs.jsonl       # triplas (context, pergunta, resposta)
│   │   └── react_cycles.jsonl    # ciclos ReAct completos sintéticos
│   └── processed/
│       └── train.jsonl           # dataset final em formato ChatML
├── src/
│   ├── generate_dataset.py       # ChromaDB → pares sintéticos via Ollama
│   ├── train.py                  # QLoRA com Unsloth, salva adapter LoRA
│   ├── export.py                 # adapter + base → GGUF → ollama create
│   └── evaluate.py               # roda evals da Fase 2 no modelo fine-tuned
└── tests/
    ├── test_generate_dataset.py
    ├── test_train.py
    └── test_export.py
```

---

## Pipeline de Dados

### Dataset RAG (`data/raw/rag_pairs.jsonl`)

**Fonte:** 4.110 chunks indexados no ChromaDB (Fase 1, `fase1-rag/data/chroma_db/`)

**Geração** via `generate_dataset.py`:
1. Iterar sobre chunks do ChromaDB
2. Para cada chunk, enviar ao llama3.2 via Ollama: gerar pergunta plausível + resposta
   ideal baseada **exclusivamente** no conteúdo do chunk
3. Filtrar pares inválidos: resposta < 50 chars, pergunta vaga (< 10 chars)
4. Meta: ~500-800 pares de alta qualidade

**Formato ChatML:**
```json
{"messages": [
  {"role": "system", "content": "You are a helpful research assistant..."},
  {"role": "user", "content": "Context: [chunk]\n\nQuestion: [pergunta]"},
  {"role": "assistant", "content": "[resposta ideal]"}
]}
```

### Dataset ReAct (`data/raw/react_cycles.jsonl`)

**Fonte:** Perguntas do dataset da Fase 2 (`fase2-evals/data/eval_dataset.json`) como seed,
ampliadas com variações

**Geração** via `generate_dataset.py`:
1. Executar o agente da Fase 3 com as perguntas seed
2. Capturar ciclos onde o formato Thought→Action→Observation→Final Answer está correto
3. Completar com ciclos gerados sinteticamente (template-based) para cobrir casos edge
4. Meta: ~200-300 ciclos bem-formados

**Formato ChatML (multi-turn):**
```json
{"messages": [
  {"role": "system", "content": "[SYSTEM_PROMPT da Fase 3]"},
  {"role": "user", "content": "[pergunta]"},
  {"role": "assistant", "content": "Thought: ...\nAction: search_papers\nAction Input: ...\nObservation: ...\nFinal Answer: ..."}
]}
```

### Pré-processamento (`data/processed/train.jsonl`)

- Merge RAG + ReAct: ~60% RAG, ~40% ReAct
- Split: 90% treino, 10% validação
- Shuffle com seed fixo para reproducibilidade

---

## Treinamento QLoRA (`src/train.py`)

**Stack:** `unsloth`, `trl.SFTTrainer`, `datasets` (HuggingFace)

**Hiperparâmetros iniciais:**
```python
# QLoRA config
load_in_4bit = True
r = 16                  # LoRA rank
lora_alpha = 32
target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

# Treinamento
per_device_train_batch_size = 4
gradient_accumulation_steps = 4   # effective batch = 16
learning_rate = 2e-4
num_train_epochs = 3
max_seq_length = 2048
```

**Saída:** `output/lora_adapter/` (adapter LoRA salvo em disco)

---

## Export (`src/export.py`)

1. Merge adapter LoRA + modelo base → modelo completo (via Unsloth `save_pretrained_merged`)
2. Converter para GGUF Q4_K_M via `llama.cpp` (`convert_hf_to_gguf.py`)
3. Registrar no Ollama: `ollama create llama3.2-finetuned -f Modelfile`
4. Smoke test: `ollama run llama3.2-finetuned "teste"`

---

## Avaliação (`src/evaluate.py`)

Reutiliza o sistema de evals da Fase 2 importando `Evaluator` diretamente de
`fase2-evals/src/evaluator.py` (padrão já usado na Fase 3 com o pipeline RAG):

```python
# evaluate.py — importa Evaluator da Fase 2
import sys
sys.path.insert(0, "../fase2-evals/src")
from evaluator import Evaluator

evaluator = Evaluator(model_name="llama3.2-finetuned", ...)
result = evaluator.run()
# Compara com baseline salvo em fase2-evals/data/results/eval_fase1_*.json
```

Gera relatório comparativo: baseline vs. fine-tuned para as 3 métricas-alvo.

---

## Testes

Padrão das fases anteriores: 100% mocked nos unit tests.

| Arquivo | O que testa |
|---|---|
| `test_generate_dataset.py` | Mock ChromaDB + Ollama; verifica estrutura dos pares, filtros de qualidade |
| `test_train.py` | Mock `FastLanguageModel`, `SFTTrainer`; verifica hiperparâmetros e save do adapter |
| `test_export.py` | Mock subprocess (llama.cpp), mock ollama CLI; verifica fluxo de export |

Integration tests marcados com `@pytest.mark.integration` para os fluxos reais.

---

## Dependências

```toml
# pyproject.toml
dependencies = [
    "unsloth>=2024.11",
    "trl>=0.12",
    "datasets>=3.0",
    "peft>=0.14",
    "transformers>=4.46",
    "torch>=2.5",
    "chromadb>=0.5",    # para gerar dataset (reusa Fase 1)
    "ollama>=0.3",      # para geração sintética e smoke test
    "pytest>=8.0",
    "ruff>=0.8",
]
```

---

## Fluxo End-to-End

```
ChromaDB (Fase 1)
    ↓
generate_dataset.py → data/raw/ → data/processed/train.jsonl
    ↓
train.py → output/lora_adapter/
    ↓
export.py → output/merged/ → output/model.gguf → Ollama (llama3.2-finetuned)
    ↓
evaluate.py → relatório comparativo (baseline vs. fine-tuned)
```

---

## Não está no escopo

- Múltiplos experimentos com hiperparâmetros (sem hyperparameter sweep)
- Comparação entre modelos base (só llama3.2 3B)
- Deployment/serving além do Ollama local
- Memória persistente entre sessões do agente (limitação da Fase 3, não atacada aqui)
