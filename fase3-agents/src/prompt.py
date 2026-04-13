"""Templates de prompt para o agente ReAct.

Define o formato estrito de raciocínio Thought/Action/Observation que
o LLM deve seguir, e a função que monta o prompt completo com histórico.

Formato esperado na saída do LLM:
    Thought: <raciocínio sobre o que fazer>
    Action: <nome_da_ferramenta>
    Action Input: <argumento para a ferramenta>

ou, quando a resposta já é conhecida:
    Thought: <raciocínio final>
    Final Answer: <resposta completa>
"""

SYSTEM_PROMPT = """Você é um assistente que responde perguntas EXCLUSIVAMENTE com base \
em documentos recuperados pelo search_papers. Você NÃO tem conhecimento próprio — \
tudo o que sabe vem das buscas.

REGRAS ABSOLUTAS (violá-las é um erro grave):
1. NUNCA escreva Final Answer sem ter feito ao menos uma chamada a search_papers.
2. NUNCA use conhecimento próprio — apenas o conteúdo das Observations.
3. Se a primeira busca não bastar, chame search_papers novamente \
com query diferente.
4. Final Answer só é permitido após pelo menos uma Observation de search_papers.
5. Mesmo que "saiba" a resposta, DEVE buscar primeiro — sua memória pode \
estar errada.

FORMATO OBRIGATÓRIO — siga exatamente, sem variações:

Thought: <seu raciocínio sobre o que buscar>
Action: search_papers
Action Input: <query de busca em linguagem natural>

Após receber a Observation, você pode:
- Buscar novamente se precisar de mais informação:
  Thought: <raciocínio>
  Action: search_papers
  Action Input: <nova query>

- Ou responder se a Observation for suficiente:
  Thought: <raciocínio baseado apenas nas Observations recebidas>
  Final Answer: <resposta construída SOMENTE com o conteúdo das Observations>

Outras ferramentas disponíveis (use apenas quando necessário):
- calculate(expression): calcula expressões matemáticas (ex: "2 + 3 * 4")
- summarize(text): resume um texto longo

LEMBRETE: Action Input deve ser apenas o argumento, sem aspas extras.
"""

_OBSERVATION_TEMPLATE = "Observation: {observation}"

_STEP_TEMPLATE = (
    "Thought: {thought}\n"
    "Action: {action}\n"
    "Action Input: {action_input}\n"
    "Observation: {observation}"
)


def build_react_prompt(question: str, history: list[dict]) -> str:
    """Monta a mensagem de usuário com a pergunta e o histórico de passos.

    Não inclui o SYSTEM_PROMPT aqui — ele é enviado separadamente no
    papel 'system' do Ollama via _call_llm, onde tem mais autoridade.
    Quando não há histórico (primeiro passo), injeta instrução explícita
    para forçar a primeira ação a ser search_papers.

    Args:
        question: pergunta original do usuário.
        history: lista de dicts com chaves 'thought', 'action',
            'action_input' e 'observation'.

    Returns:
        String com a pergunta e o histórico pronta para o papel 'user'.
    """
    parts = [f"Pergunta: {question}\n"]

    for step in history:
        if step.get("observation") is not None:
            parts.append(
                _STEP_TEMPLATE.format(
                    thought=step["thought"],
                    action=step["action"],
                    action_input=step["action_input"],
                    observation=step["observation"],
                )
            )

    # Primeiro passo: instrução explícita para não pular direto para Final Answer
    if not history:
        parts.append(
            "IMPORTANTE: você ainda não fez nenhuma busca. "
            "Sua primeira ação OBRIGATÓRIA é:\n"
            "Thought: <raciocínio sobre o que buscar>\n"
            "Action: search_papers\n"
            "Action Input: <query>"
        )

    return "\n".join(parts)
