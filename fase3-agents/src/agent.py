"""Loop ReAct principal — Reasoning + Acting.

Implementa o agente do zero, sem frameworks externos. O loop funciona assim:
1. Monta o prompt com o histórico atual
2. Chama o LLM e obtém a saída
3. Faz parse para identificar Action ou Final Answer
4. Se Action: executa a ferramenta e adiciona Observation ao histórico
5. Se Final Answer: retorna o resultado
6. Se atingiu max_steps: retorna com truncated=True

Referência: "ReAct: Synergizing Reasoning and Acting in Language Models"
(Yao et al., 2022)
"""

import logging
import re
from dataclasses import dataclass, field

import ollama

from src.prompt import SYSTEM_PROMPT, build_react_prompt

logger = logging.getLogger(__name__)

# Padrões regex para parse da saída do LLM
_RE_THOUGHT = re.compile(r"Thought:\s*(.+?)(?=\n(?:Action|Final Answer)|$)", re.DOTALL)
_RE_ACTION = re.compile(r"Action:\s*(\w+)")
_RE_ACTION_INPUT = re.compile(
    r"Action Input:\s*(.+?)(?=\nObservation|\nThought|$)", re.DOTALL
)
_RE_FINAL_ANSWER = re.compile(r"Final Answer:\s*(.+?)$", re.DOTALL)


@dataclass
class Step:
    """Representa um passo do raciocínio do agente."""

    thought: str
    action: str
    action_input: str
    observation: str


@dataclass
class AgentResult:
    """Resultado completo de uma execução do agente."""

    answer: str
    steps: list[Step] = field(default_factory=list)
    truncated: bool = False
    total_steps: int = 0


def _parse_llm_output(text: str) -> dict:
    """Faz parse da saída do LLM para extrair os campos ReAct.

    Extrai Thought, Action/Action Input ou Final Answer do texto gerado.
    Retorna um dict com as chaves identificadas. Se o formato estiver
    malformado, retorna os campos disponíveis com strings vazias nos demais.

    Args:
        text: saída bruta do LLM.

    Returns:
        Dict com subset de: 'thought', 'action', 'action_input', 'final_answer'.
    """
    result: dict = {}

    thought_match = _RE_THOUGHT.search(text)
    if thought_match:
        result["thought"] = thought_match.group(1).strip()

    final_match = _RE_FINAL_ANSWER.search(text)
    if final_match:
        result["final_answer"] = final_match.group(1).strip()
        return result

    action_match = _RE_ACTION.search(text)
    input_match = _RE_ACTION_INPUT.search(text)
    if action_match:
        result["action"] = action_match.group(1).strip()
        result["action_input"] = input_match.group(1).strip() if input_match else ""

    return result


class ReActAgent:
    """Agente ReAct que usa ferramentas para responder perguntas multi-hop.

    O agente alterna entre raciocínio (Thought) e ação (Action) até
    chegar a uma resposta final ou esgotar o limite de passos.

    Args:
        tools: dict mapeando nome da ferramenta para a função callable.
        model: nome do modelo Ollama para geração.
        max_steps: número máximo de iterações antes de truncar.
    """

    def __init__(
        self,
        tools: dict,
        model: str = "llama3.2",
        max_steps: int = 10,
    ) -> None:
        self.tools = tools
        self.model = model
        self.max_steps = max_steps

    def _call_llm(self, user_content: str) -> str:
        """Chama o LLM via Ollama com SYSTEM_PROMPT no papel 'system'.

        Separar system de user dá muito mais peso às instruções do sistema
        — o modelo respeita restrições de papel 'system' com muito mais
        consistência do que quando as mesmas instruções chegam como 'user'.

        Args:
            user_content: pergunta + histórico montados por build_react_prompt.

        Returns:
            Resposta gerada pelo modelo.
        """
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        return response.message.content

    def _execute_tool(self, action: str, action_input: str) -> str:
        """Executa a ferramenta solicitada e retorna o resultado.

        Args:
            action: nome da ferramenta (deve estar em self.tools).
            action_input: argumento a passar para a ferramenta.

        Returns:
            Resultado da ferramenta como string, ou mensagem de erro
            se a ferramenta não existir ou lançar exceção.
        """
        tool_fn = self.tools.get(action)
        if tool_fn is None:
            available = ", ".join(self.tools.keys())
            return f"Ferramenta '{action}' não encontrada. Disponíveis: {available}"
        try:
            result = tool_fn(action_input)
            # search_papers retorna lista — formata para texto
            if isinstance(result, list):
                if not result:
                    return "Nenhum resultado encontrado."
                parts = []
                for i, item in enumerate(result, 1):
                    content = item.get("content", "")
                    meta = item.get("metadata", {})
                    source = meta.get("source", "?")
                    page = meta.get("page", 0)
                    parts.append(f"[{i}] ({source}, p.{page})\n{content}")
                return "\n\n".join(parts)
            return str(result)
        except Exception as e:  # noqa: BLE001
            return f"Erro ao executar '{action}': {e}"

    def run(self, question: str) -> AgentResult:
        """Executa o loop ReAct para responder a uma pergunta.

        Itera até encontrar um Final Answer ou atingir max_steps.
        Cada passo é registrado no log e adicionado ao histórico.

        Args:
            question: pergunta a ser respondida pelo agente.

        Returns:
            AgentResult com a resposta, lista de steps e flag de truncamento.
        """
        logger.info("Iniciando agente para: %r", question)
        history: list[dict] = []
        steps: list[Step] = []

        for step_num in range(1, self.max_steps + 1):
            logger.info("--- Passo %d/%d ---", step_num, self.max_steps)

            prompt = build_react_prompt(question, history)
            raw_output = self._call_llm(prompt)
            logger.debug("Saída bruta do LLM:\n%s", raw_output)

            parsed = _parse_llm_output(raw_output)
            thought = parsed.get("thought", "")

            # Verifica se o LLM chegou a uma resposta final
            if "final_answer" in parsed:
                logger.info("Final Answer encontrado no passo %d", step_num)
                return AgentResult(
                    answer=parsed["final_answer"],
                    steps=steps,
                    truncated=False,
                    total_steps=step_num,
                )

            # Saída malformada — não tem action nem final_answer
            action = parsed.get("action", "")
            action_input = parsed.get("action_input", "")
            if not action:
                logger.warning(
                    "Saída malformada no passo %d — sem Action nem Final Answer",
                    step_num,
                )
                observation = (
                    "Formato inválido. Use Action: <ferramenta> "
                    "ou Final Answer: <resposta>."
                )
            else:
                logger.info("Action=%r Input=%r", action, action_input)
                observation = self._execute_tool(action, action_input)
                logger.info("Observation=%r", observation[:100])

            step = Step(
                thought=thought,
                action=action,
                action_input=action_input,
                observation=observation,
            )
            steps.append(step)

            history.append(
                {
                    "thought": thought,
                    "action": action,
                    "action_input": action_input,
                    "observation": observation,
                }
            )

        # Esgotou max_steps sem Final Answer
        logger.warning("Agente truncado após %d passos", self.max_steps)
        best_thought = steps[-1].thought if steps else "Sem informação suficiente."
        return AgentResult(
            answer=best_thought,
            steps=steps,
            truncated=True,
            total_steps=self.max_steps,
        )
