# Próximos Passos — Fase 4: Fine-tuning

## O que foi construído na Fase 3

Agente ReAct implementado do zero, sem frameworks externos, usando o pipeline RAG da Fase 1
como ferramenta de busca para responder perguntas multi-hop.

| Módulo | Função | Status |
|---|---|---|
| `tools.py` | `search_papers` (ChromaDB), `calculate` (AST seguro), `summarize` | Completo |
| `prompt.py` | `SYSTEM_PROMPT` no papel `system`, `build_react_prompt` com histórico | Completo |
| `agent.py` | `ReActAgent`, loop ReAct, parse via regex, `AgentResult`, `Step` | Completo |
| `run_agent.py` | CLI `--question` e modo interativo | Completo |

**Métricas da Fase 3:**
- 30 testes unitários, todos passando, todos mockados
- Loop Thought → Action → Observation → Final Answer confirmado em testes reais
- SYSTEM_PROMPT no papel `system` do Ollama — eliminado o problema de resposta sem busca

---

## Limitações identificadas na Fase 3

1. **llama3.2 (3B) segue o formato ReAct de forma inconsistente** — polui `Action Input`
   com texto extra, às vezes mistura Thought com o argumento da ferramenta. Modelos maiores
   seguiriam o formato estrito com muito mais consistência.
2. **Corpus do ArXiv não cobre perguntas sobre RAG em si** — o retrieval recupera chunks
   irrelevantes para perguntas sobre o próprio sistema. Limitação do corpus, não do agente.
3. **Faithfulness e Answer Relevance abaixo do ideal** — herdados do baseline da Fase 2
   (0.57 e 0.60). O agente não piora, mas também não melhora a qualidade do modelo base.
4. **Parsing frágil** — regex funciona para llama3.2 mas é sensível a variações de formato.
   Structured outputs ou validação com pydantic seria mais robusto.
5. **Sem memória entre sessões** — o histórico de raciocínio existe apenas durante uma
   execução. Para um agente persistente seria necessário serializar o histórico.

---

## O que será construído na Fase 4 — Fine-tuning

### Objetivo principal

Fazer fine-tuning de um modelo pequeno para melhorar dois problemas concretos identificados
nas fases anteriores:

1. **Qualidade das respostas RAG** — aumentar Faithfulness > 0.70 e Answer Relevance > 0.70
   (baseline: 0.57 e 0.60 da Fase 2)
2. **Aderência ao formato ReAct** — o modelo fine-tunado deve seguir o formato
   Thought/Action/Action Input sem poluição ou extrapolação

### Por que fine-tuning e não só um prompt melhor?

Prompt engineering tem um teto. O llama3.2 (3B) tem capacidade limitada de seguir
instruções complexas e formatos estritos — não porque o prompt é ruim, mas porque o modelo
não foi treinado para isso. Fine-tuning ensina o modelo com exemplos concretos do
comportamento desejado, em vez de apenas descrever em texto o que se espera.

### Dataset de fine-tuning

Gerar pares de treinamento a partir do corpus já existente (4.110 chunks do ArXiv):

**Para qualidade RAG:** pares `(pergunta, resposta ideal baseada no chunk)`
```json
{
  "instruction": "Responda com base apenas no contexto fornecido.",
  "input": "Contexto: [chunk]\nPergunta: [pergunta gerada pelo LLM]",
  "output": "[resposta fiel ao chunk, sem extrapolação]"
}
```

**Para formato ReAct:** exemplos de ciclos completos Thought → Action → Observation → Final Answer
com o formato correto como saída esperada.

### Técnica: LoRA / QLoRA

Fine-tuning completo de um LLM exige dezenas de GB de VRAM. LoRA (Low-Rank Adaptation)
congela os pesos originais e treina apenas matrizes de adaptação de baixa dimensão —
reduz o custo de memória em ~10x com perda mínima de qualidade.

QLoRA adiciona quantização (4-bit) ao modelo base durante o treino — permite fine-tuning
de um modelo 7B em uma GPU consumer de 8GB VRAM (ex: RTX 3080/4070).

### Stack planejada

- **Modelo base:** llama3.2 (3B) — manter consistência com as fases anteriores, ou
  Mistral 7B se os resultados com 3B forem insuficientes
- **Framework de treino:** [Unsloth](https://github.com/unslothai/unsloth) (2x mais rápido
  que TRL padrão) ou HuggingFace TRL com PEFT
- **Formato do dataset:** Alpaca (instruction/input/output) ou ChatML
- **Exportação:** GGUF via `llama.cpp` para servir com Ollama localmente
- **Avaliação:** sistema de evals da Fase 2 rodando sobre o modelo fine-tunado

### Métricas de sucesso

| Métrica | Baseline (Fase 2) | Meta Fase 4 |
|---|---|---|
| Faithfulness médio | 0.57 | > 0.70 |
| Answer Relevance médio | 0.60 | > 0.70 |
| Formato ReAct correto (%) | ~70% | > 90% |

### Arquitetura planejada

```
fase4-finetuning/
├── pyproject.toml
├── data/
│   ├── raw/                   # pares brutos gerados pelo pipeline RAG
│   └── processed/             # dataset final no formato de treino
├── src/
│   ├── generate_dataset.py    # usa ChromaDB + LLM para gerar exemplos
│   ├── train.py               # loop de fine-tuning com LoRA/QLoRA
│   ├── export.py              # converte adaptador para GGUF e registra no Ollama
│   └── evaluate.py            # roda os evals da Fase 2 no modelo fine-tunado
└── tests/
    └── test_generate_dataset.py
```

### Fluxo completo da Fase 4

```
1. Gerar dataset
   ChromaDB (chunks) → LLM gera perguntas → valida pares → salva em data/processed/

2. Fine-tuning
   Modelo base + dataset → LoRA → adaptador treinado

3. Exportar
   Adaptador + modelo base → merge → GGUF → ollama create fase4-model

4. Avaliar
   Sistema de evals da Fase 2 rodando com fase4-model → comparar com baseline
```
