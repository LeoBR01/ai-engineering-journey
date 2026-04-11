# CLAUDE.md — Regras do Projeto RAG

## Visão Geral

Pipeline RAG (Retrieval-Augmented Generation) 100% local.
LLM via Ollama, vector store via ChromaDB, leitura de PDFs via PyMuPDF.

## Gerenciamento de Ambiente

- Usar **uv** para tudo: `uv sync`, `uv run`, `uv add`
- Nunca usar `pip install` diretamente
- Dependências declaradas em `pyproject.toml`

## Lint e Formatação

- Usar **ruff** para lint e formatação
- Antes de qualquer commit: `ruff check . && ruff format .`
- Não ignorar erros de lint sem justificativa explícita no código

## Estrutura de Módulos

Cada módulo em `src/` tem responsabilidade única:

| Arquivo | Responsabilidade |
|---|---|
| `ingestion.py` | Leitura de PDFs e chunking de texto |
| `embeddings.py` | Geração de embeddings e persistência no ChromaDB |
| `retrieval.py` | Busca semântica de chunks relevantes |
| `generation.py` | Geração de resposta via Ollama com contexto |
| `pipeline.py` | Orquestração do fluxo completo |

## Regras de Código

- Todas as funções públicas devem ter docstrings
- Type hints obrigatórios em todas as assinaturas de funções
- Sem lógica de negócio em `pipeline.py` — apenas orquestração
- PDFs ficam em `data/pdfs/`; nunca commitar PDFs no git
- Configurações (modelo Ollama, collection name, chunk size) como constantes no topo dos módulos ou via variável de ambiente

## Testes

- Um arquivo de teste por módulo em `tests/`
- Usar `pytest`
- Mocks para Ollama e ChromaDB nos testes unitários (evitar dependências externas)

## Documentação no Obsidian

Após QUALQUER alteração significativa no projeto (nova feature, decisão de arquitetura, bug resolvido, conceito implementado), você DEVE atualizar os arquivos em `/home/lbr/obsidian-vault/AI Engineering Journey/Fase 1 - RAG/`

Arquivos a manter atualizados:
- `01 - Fundamentos e Decisões.md` → decisões de arquitetura e por quê cada escolha foi feita
- `02 - O que Aprendi.md` → conceitos explicados de forma didática, como se ensinasse alguém
- `03 - Problemas e Soluções.md` → erros encontrados e como foram resolvidos

Regras:
- Escreva sempre em português
- Linguagem didática, não só técnica — explique o PORQUÊ, não só o O QUÊ
- Nunca sobrescreva conteúdo existente, sempre ACRESCENTE com data
- **NUNCA usar o MCP do Obsidian** para escrever nos arquivos — ele causa timeout. Sempre usar Read/Edit/Write diretamente no filesystem nos caminhos acima
- **Após qualquer atualização no Obsidian**, rodar automaticamente: `~/sync-obsidian.sh`
