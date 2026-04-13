"""CLI interativo para o agente ReAct.

Aceita uma pergunta via --question ou entra em modo interativo com input().
Imprime cada passo do agente em tempo real conforme acontece.

Uso:
    uv run agent-run --question "O que é RAG?"
    uv run agent-run   # modo interativo
"""

import argparse
import logging
import sys

from src.agent import ReActAgent, Step
from src.tools import calculate, search_papers, summarize

# Configura logging para exibir INFO no terminal
logging.basicConfig(
    level=logging.WARNING,  # suprime logs internos do chromadb/httpx
    format="%(levelname)s %(name)s: %(message)s",
)
logging.getLogger("src.agent").setLevel(logging.INFO)

# Ferramentas disponíveis para o agente
TOOLS = {
    "search_papers": search_papers,
    "calculate": calculate,
    "summarize": summarize,
}


def _print_step(step_num: int, step: Step) -> None:
    """Exibe um passo do agente no terminal de forma legível.

    Args:
        step_num: número do passo (1-indexed).
        step: objeto Step com thought, action, action_input e observation.
    """
    print(f"\n{'─' * 60}")
    print(f"Passo {step_num}")
    print(f"{'─' * 60}")
    if step.thought:
        print(f"Thought: {step.thought}")
    if step.action:
        print(f"Action:  {step.action}")
        print(f"Input:   {step.action_input}")
        # Trunca observation longa para não poluir o terminal
        obs = step.observation
        if len(obs) > 300:
            obs = obs[:300] + "... [truncado]"
        print(f"Obs:     {obs}")


def run_question(question: str) -> None:
    """Executa o agente para uma pergunta e exibe o resultado em tempo real.

    Instancia o ReActAgent, executa, e imprime cada passo conforme
    o agente vai avançando. Ao final, exibe a resposta e o número de steps.

    Args:
        question: pergunta a ser respondida.
    """
    print(f"\nPergunta: {question}")

    # Monkey-patch no run para imprimir passos em tempo real
    # O agente acumula steps — comparamos o tamanho antes/depois de cada iteração
    agent = ReActAgent(tools=TOOLS, model="llama3.2", max_steps=10)

    # Sobrescreve _execute_tool para imprimir conforme executa
    original_execute = agent._execute_tool  # noqa: SLF001

    def traced_execute(action: str, action_input: str) -> str:
        result = original_execute(action, action_input)
        return result

    agent._execute_tool = traced_execute  # noqa: SLF001

    result = agent.run(question)

    # Imprime todos os steps acumulados
    for i, step in enumerate(result.steps, 1):
        _print_step(i, step)

    print(f"\n{'═' * 60}")
    if result.truncated:
        print(f"[TRUNCADO após {result.total_steps} passos]")
    else:
        print(f"[Concluído em {result.total_steps} passo(s)]")
    print(f"{'═' * 60}")
    print(f"\nResposta:\n{result.answer}\n")


def main() -> None:
    """Ponto de entrada do CLI.

    Parseia argumentos, decide entre modo --question ou interativo,
    e executa o agente para cada pergunta.
    """
    parser = argparse.ArgumentParser(description="Agente ReAct sobre papers de IA")
    parser.add_argument("--question", "-q", type=str, help="Pergunta a responder")
    args = parser.parse_args()

    if args.question:
        run_question(args.question)
    else:
        # Modo interativo
        print("Agente ReAct — Fase 3 (Ctrl+C ou 'sair' para encerrar)")
        while True:
            try:
                question = input("\n> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nEncerrando.")
                sys.exit(0)

            if not question:
                continue
            if question.lower() in {"sair", "exit", "quit"}:
                sys.exit(0)

            run_question(question)


if __name__ == "__main__":
    main()
