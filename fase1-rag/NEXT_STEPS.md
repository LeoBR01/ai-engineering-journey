# Próximos Passos — Fase 2: Evals

## O que foi construído na Fase 1

Pipeline RAG 100% local, funcional de ponta a ponta:

| Módulo | Função | Status |
|---|---|---|
| `ingestion.py` | Lê PDFs, extrai texto por página, aplica chunking com overlap | Completo |
| `embeddings.py` | Gera embeddings via `nomic-embed-text`, persiste no ChromaDB | Completo |
| `retrieval.py` | Busca semântica top-K, formata contexto para o LLM | Completo |
| `generation.py` | Monta prompt RAG, chama `llama3.2` via Ollama | Completo |
| `pipeline.py` | Orquestra os quatro módulos, expõe CLI `index`/`query` | Completo |
| `download_papers.py` | Baixa papers do ArXiv automaticamente | Completo |

**Métricas da Fase 1:**
- 38 testes unitários, todos passando, todos mockados (sem dependências externas)
- 25 PDFs indexados (papers do ArXiv sobre LLM agents, RAG, fine-tuning)
- 4.110 chunks armazenados na collection `papers` do ChromaDB
- Cobertura: ingestion, embeddings, retrieval, generation, pipeline

---

## Limitações identificadas

### 1. Sem avaliação de qualidade das respostas
O pipeline gera respostas, mas não há nenhuma métrica para saber se são boas. Uma resposta pode parecer fluente e ainda assim estar errada ou incompleta. **Não sabemos o quão bem o RAG está funcionando.**

### 2. Chunking ingênuo por caracteres
O chunking atual divide o texto em blocos de 512 caracteres independentemente da estrutura do conteúdo. Isso pode cortar frases no meio, separar títulos de seção do corpo, ou quebrar uma lista enumerada ao meio. Papers científicos têm estrutura semântica rica que está sendo ignorada.

### 3. Queries do ArXiv sem filtro de categoria
As buscas por "RAG" e "LLM agents" retornam papers de datas recentes, não necessariamente relevantes para LLMs. Papers sobre visão computacional e robótica apareceram nos resultados. Os PDFs indexados são heterogêneos.

### 4. Re-indexação não é idempotente
Rodar `index` duas vezes sobre os mesmos PDFs cria chunks duplicados no ChromaDB (novos UUIDs a cada execução). Não há verificação de documentos já indexados.

### 5. Sem tratamento de PDFs com imagens ou tabelas
PyMuPDF extrai texto linear. PDFs com equações matemáticas (comuns em papers de ML), figuras com texto embutido ou tabelas complexas perdem informação na extração.

### 6. Modelo de geração limitado
O `llama3.2` é um modelo compacto (3B parâmetros). Funciona para testes, mas tem raciocínio limitado em perguntas complexas sobre papers técnicos. Não há fallback nem suporte a múltiplos modelos.

### 7. Sem histórico de conversa
Cada query é independente. Não há memória de perguntas anteriores — não é possível fazer perguntas de acompanhamento ("e sobre o paper que você mencionou?").

---

## O que será melhorado na Fase 2 — Evals

### Objetivo principal
Construir um sistema de avaliação (evals) que permita medir objetivamente a qualidade do RAG e comparar variações de configuração.

### 1. Dataset de avaliação
Criar um conjunto de pares `(pergunta, resposta_esperada)` sobre os papers indexados. Cada pergunta deve ter uma resposta verificável no texto do paper.

```
Exemplo:
{
  "question": "Qual métrica o paper ClawBench usa para avaliar agentes?",
  "expected": "Taxa de conclusão de tarefas (task completion rate)",
  "source": "ClawBench_paper.pdf",
  "page": 4
}
```

### 2. Métricas de retrieval
Avaliar se o retrieval está encontrando os chunks certos:
- **Recall@K:** o chunk correto está entre os K retornados?
- **MRR (Mean Reciprocal Rank):** em que posição o chunk correto aparece?

### 3. Métricas de geração
Avaliar se a resposta gerada é correta e grounded:
- **Faithfulness:** a resposta está de fato suportada pelo contexto recuperado?
- **Answer relevance:** a resposta responde à pergunta feita?
- Usar um LLM como juiz (LLM-as-a-judge) para avaliar essas dimensões

### 4. Experimentos de chunking
Comparar estratégias de chunking e medir impacto nas métricas:
- Por sentença (NLTK/spaCy)
- Por parágrafo
- Chunking semântico (dividir quando o tema muda)
- Tamanhos diferentes: 256, 512, 1024 chars

### 5. Re-indexação idempotente
Gerar `chunk_id` deterministico baseado em `hash(source + page + chunk_index)` em vez de UUID aleatório. Assim `index` pode ser rodado múltiplas vezes sem duplicatas.

### 6. Filtro de categoria no ArXiv
Adicionar `cat:cs.CL OR cat:cs.AI` às queries para garantir que os papers baixados são sobre NLP/IA, não outras áreas.

### 7. Logging estruturado
Substituir os `print()` por `logging` estruturado com níveis (INFO, DEBUG). Facilita debugging e futura integração com ferramentas de observabilidade.
