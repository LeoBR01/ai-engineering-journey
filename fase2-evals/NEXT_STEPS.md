# Próximos Passos — Fase 3: Agentes

## O que foi construído na Fase 2

Sistema de avaliação quantitativa para o pipeline RAG da Fase 1, executado de ponta a ponta:

| Módulo | Função | Status |
|---|---|---|
| `dataset.py` | Dataset de 15 perguntas (easy/medium/hard) com keywords e expected answers | Completo |
| `metrics_retrieval.py` | Recall@K e MRR — avalia qualidade do retrieval | Completo |
| `metrics_generation.py` | Faithfulness e Answer Relevance via LLM-as-judge (llama3.2) | Completo |
| `evaluator.py` | Orquestrador com injeção de dependência | Completo |
| `report.py` | Formata e persiste resultados com timestamp | Completo |
| `run_eval.py` | Integração com ChromaDB e Ollama da Fase 1, tabela rich | Completo |

**Métricas da Fase 2:**
- 44 testes unitários, todos passando, todos mockados
- 15 perguntas avaliadas: 5 easy, 5 medium, 5 hard
- Cobertura: dataset, retrieval metrics, generation metrics, evaluator, run_eval

---

## Resultados do primeiro experimento (2026-04-11)

Primeiro experimento real com 15 perguntas, llama3.2 + nomic-embed-text + 4.110 chunks:

| Métrica | Valor | Observação |
|---|---|---|
| Faithfulness médio | **0.57** | Modelo às vezes vai além do contexto recuperado |
| Answer Relevance médio | **0.60** | Respostas nem sempre respondem diretamente à pergunta |
| Recall@5 médio | não coletado | ChromaDB não expôs documentos relevantes ground-truth |
| MRR médio | não coletado | Requer anotação manual de `relevant_chunk_ids` |

- **Perguntas com melhor performance:** q001, q003, q005, q007, q009, q011, q015 — Faithfulness 0.80
- **Perguntas com pior performance:** q002 (Answer Relevance 0.00), q004 e q012 (Faithfulness 0.00)
- **Insight principal:** perguntas hard sobre trade-offs tiveram Answer Relevance 0.72 vs 0.58 das
  easy — o corpus de papers técnicos favorece discussões de limitações, não definições básicas

---

## Limitações identificadas na Fase 2

1. **Dataset pequeno (15 perguntas)** — não é estatisticamente significativo para comparar estratégias
2. **LLM-as-judge usando o mesmo modelo que gera** — viés de auto-avaliação; em produção usar modelo maior
3. **Recall@5/MRR não anotados** — exige anotação manual dos chunk IDs corretos por pergunta
4. **Keywords como proxy são frágeis** — um chunk pode ser relevante sem conter a keyword exata
5. **Experimentos de chunking planejados mas não executados** — variações de tamanho (256/512/1024) e estratégia (sentença, parágrafo, semântico)
6. **Sem análise das respostas geradas** — o relatório armazena as respostas mas não há ferramenta para inspecioná-las facilmente

---

## O que será construído na Fase 3 — Agentes

### Objetivo principal

Construir um agente ReAct (Reasoning + Acting) que usa o pipeline RAG da Fase 1 como
ferramenta de busca. O agente deve resolver perguntas multi-hop que exigem múltiplas
buscas e raciocínio entre os resultados.

### Ferramentas do agente

```python
tools = {
    "search_papers": search_papers,    # chama retrieve() da Fase 1
    "calculate": calculate,            # avalia expressões matemáticas simples
    "summarize": summarize,            # resume um trecho longo via LLM
}
```

### Loop ReAct

O agente itera em ciclos Thought → Action → Observation até ter a resposta final:

```
Thought: preciso buscar sobre RAG para responder essa pergunta
Action: search_papers("retrieval augmented generation limitations")
Observation: [chunks retornados]
Thought: os chunks mencionam o problema de context window, vou buscar mais
Action: search_papers("context window lost in the middle problem")
Observation: [mais chunks]
Thought: tenho informação suficiente para responder
Final Answer: ...
```

### Stack

- **Python puro** — implementar o loop ReAct do zero, sem LangChain ou frameworks
- **Ollama + llama3.2** — modelo do agente (mesmo da Fase 1 e 2)
- **ChromaDB da Fase 1** — knowledge base de papers do ArXiv
- **Estrutura de projeto** idêntica às fases anteriores (uv, ruff, pytest)

### Métricas de sucesso

- O agente consegue responder perguntas multi-hop que exigem mais de uma busca
- O agente para quando tem a resposta (não fica em loop infinito) — max_steps configurável
- Logs estruturados de cada passo Thought/Action/Observation para debugging
- Os evals da Fase 2 continuam funcionando para medir a qualidade das respostas do agente

### Arquitetura planejada

```
fase3-agents/
├── src/
│   ├── tools.py          # search_papers, calculate, summarize
│   ├── agent.py          # loop ReAct principal
│   ├── prompt.py         # templates de prompt para o agente
│   └── run_agent.py      # CLI para interagir com o agente
└── tests/
    ├── test_tools.py
    └── test_agent.py
```
