# CLAUDE.md — Regras do Projeto Evals (Fase 2)

## Visão Geral

Sistema de avaliação quantitativa para o pipeline RAG da Fase 1.
Mede qualidade de retrieval (Recall@K, MRR) e geração (Faithfulness, Answer Relevance).

## Dependência da Fase 1

Este projeto **não reimplementa** o RAG — importa diretamente os módulos de
`../fase1-rag/src/`. O ChromaDB da Fase 1 (`../fase1-rag/chroma_db/`) é usado
como fonte de dados para os experimentos.

## Gerenciamento de Ambiente

- Usar **uv** para tudo: `uv sync`, `uv run`, `uv add`
- Nunca usar `pip install` diretamente

## Lint e Formatação

- Usar **ruff** para lint e formatação
- Antes de qualquer commit: `ruff check . && ruff format .`

## Estrutura de Módulos

| Arquivo | Responsabilidade |
|---|---|
| `dataset.py` | Criação, carregamento e validação do dataset de avaliação |
| `metrics_retrieval.py` | Recall@K e MRR — avalia qualidade do retrieval |
| `metrics_generation.py` | Faithfulness e Answer Relevance via LLM-as-judge |
| `evaluator.py` | Orquestra a avaliação completa sobre o dataset |
| `report.py` | Formata e persiste os resultados em `data/results/` |

## Formato do Dataset

Arquivo `data/eval_dataset.json` — lista de objetos:

```json
{
  "id": "q001",
  "question": "Qual métrica o paper X usa para avaliar agentes?",
  "expected_answer": "Taxa de conclusão de tarefas",
  "source_document": "nome_do_paper.pdf",
  "relevant_chunk_ids": ["chunk_hash_1", "chunk_hash_2"]
}
```

## Regras de Código

- Todas as funções públicas devem ter docstrings
- Type hints obrigatórios em todas as assinaturas
- Sem dependências de I/O nos módulos de métricas (recebem dados via parâmetros)
- LLM-as-judge usa `llama3.2` via Ollama (mesmo modelo da Fase 1)

## Testes

- Um arquivo de teste por módulo em `tests/`
- Mocks para ChromaDB e Ollama — sem dependências externas nos testes unitários
- Usar `pytest`

## Documentação no Obsidian

Após QUALQUER alteração significativa, atualizar:
`/home/lbr/obsidian-vault/AI Engineering Journey/Fase 2 - Evals/`

Arquivos a manter:
- `01 - Fundamentos e Decisões.md` → decisões de arquitetura
- `02 - O que Aprendi.md` → conceitos explicados didaticamente
- `03 - Problemas e Soluções.md` → erros encontrados e resoluções

Regras:
- Sempre em português, linguagem didática
- Nunca sobrescrever — sempre acrescentar com data
- **NUNCA usar MCP do Obsidian** — escrever direto no filesystem (Read/Edit/Write)
- **Após cada atualização**, rodar automaticamente: `~/sync-obsidian.sh`
