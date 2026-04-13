"""Testes unitários para agent.py — LLM e tools totalmente mockados."""

from unittest.mock import MagicMock, patch

from src.agent import AgentResult, ReActAgent, Step, _parse_llm_output

# ── _parse_llm_output ─────────────────────────────────────────────────────────


def test_parse_extrai_thought_e_action():
    """_parse_llm_output deve extrair Thought, Action e Action Input."""
    text = (
        "Thought: Preciso buscar sobre RAG.\n"
        "Action: search_papers\n"
        "Action Input: O que é RAG?"
    )
    result = _parse_llm_output(text)
    assert result["thought"] == "Preciso buscar sobre RAG."
    assert result["action"] == "search_papers"
    assert result["action_input"] == "O que é RAG?"


def test_parse_extrai_final_answer():
    """_parse_llm_output deve extrair Final Answer quando presente."""
    text = (
        "Thought: Já tenho informação suficiente.\n"
        "Final Answer: RAG combina retrieval com geração de texto."
    )
    result = _parse_llm_output(text)
    assert result["final_answer"] == "RAG combina retrieval com geração de texto."
    assert "action" not in result


def test_parse_saida_malformada_retorna_dict_parcial():
    """_parse_llm_output deve retornar dict parcial para saída malformada."""
    result = _parse_llm_output("Texto completamente fora do formato esperado.")
    # Não deve lançar exceção — retorna o que conseguir
    assert isinstance(result, dict)
    assert "final_answer" not in result
    assert "action" not in result


def test_parse_final_answer_multiline():
    """_parse_llm_output deve capturar Final Answer com múltiplas linhas."""
    text = "Thought: Tenho tudo.\nFinal Answer: Linha um.\nLinha dois.\nLinha três."
    result = _parse_llm_output(text)
    assert "Linha um." in result["final_answer"]
    assert "Linha dois." in result["final_answer"]


# ── ReActAgent — ciclo completo ───────────────────────────────────────────────


def _make_agent(tool_fn=None, max_steps=10):
    """Cria um ReActAgent com ferramentas mockadas."""
    tools = {"search_papers": tool_fn or MagicMock(return_value=[])}
    return ReActAgent(tools=tools, model="llama3.2", max_steps=max_steps)


@patch("src.agent.ollama.chat")
def test_agente_ciclo_completo_dois_passos(mock_chat):
    """Agente deve seguir Thought→Action→Observation→Final Answer em 2 passos."""
    # Passo 1: LLM decide buscar
    resp1 = MagicMock()
    resp1.message.content = (
        "Thought: Vou buscar informações sobre RAG.\n"
        "Action: search_papers\n"
        "Action Input: RAG retrieval augmented generation"
    )
    # Passo 2: LLM dá a resposta final
    resp2 = MagicMock()
    resp2.message.content = (
        "Thought: Encontrei a informação necessária.\n"
        "Final Answer: RAG é uma técnica que combina busca semântica com geração de texto."  # noqa: E501
    )
    mock_chat.side_effect = [resp1, resp2]

    tool_fn = MagicMock(
        return_value=[
            {"content": "RAG info", "metadata": {"source": "p.pdf", "page": 1}}
        ]
    )
    agent = _make_agent(tool_fn=tool_fn)

    result = agent.run("O que é RAG?")

    assert isinstance(result, AgentResult)
    assert not result.truncated
    assert result.total_steps == 2
    assert len(result.steps) == 1  # só o step com action/observation
    assert "RAG" in result.answer
    tool_fn.assert_called_once_with("RAG retrieval augmented generation")


@patch("src.agent.ollama.chat")
def test_agente_resposta_direta_sem_ferramenta(mock_chat):
    """Agente deve retornar Final Answer imediatamente se não precisar de tools."""
    resp = MagicMock()
    resp.message.content = (
        "Thought: Sei a resposta diretamente.\nFinal Answer: 2 + 2 = 4"
    )
    mock_chat.return_value = resp

    agent = _make_agent()
    result = agent.run("Quanto é 2 + 2?")

    assert not result.truncated
    assert result.total_steps == 1
    assert result.steps == []  # nenhuma action executada
    assert "4" in result.answer


@patch("src.agent.ollama.chat")
def test_agente_truncado_ao_atingir_max_steps(mock_chat):
    """Agente deve retornar truncated=True se esgotar max_steps sem Final Answer."""
    resp = MagicMock()
    resp.message.content = (
        "Thought: Ainda buscando...\n"
        "Action: search_papers\n"
        "Action Input: mais informação"
    )
    mock_chat.return_value = resp  # sempre retorna Action, nunca Final Answer

    agent = _make_agent(max_steps=3)
    result = agent.run("Pergunta impossível")

    assert result.truncated is True
    assert result.total_steps == 3
    assert len(result.steps) == 3


@patch("src.agent.ollama.chat")
def test_agente_saida_malformada_continua_sem_travar(mock_chat):
    """Agente deve continuar o loop mesmo se o LLM retornar saída malformada."""
    # Passos 1 e 2: saída inválida
    resp_bad = MagicMock()
    resp_bad.message.content = "Isso não segue o formato."
    # Passo 3: resposta válida
    resp_good = MagicMock()
    resp_good.message.content = (
        "Thought: Agora entendi.\nFinal Answer: Aqui está a resposta."
    )
    mock_chat.side_effect = [resp_bad, resp_bad, resp_good]

    agent = _make_agent(max_steps=10)
    result = agent.run("Pergunta qualquer")

    assert not result.truncated
    assert result.total_steps == 3
    assert "Aqui está a resposta." in result.answer


@patch("src.agent.ollama.chat")
def test_agente_ferramenta_inexistente_recebe_mensagem_de_erro(mock_chat):
    """Agente deve retornar mensagem de erro na Observation para tool inexistente."""
    resp1 = MagicMock()
    resp1.message.content = (
        "Thought: Tentarei usar uma ferramenta que não existe.\n"
        "Action: ferramenta_fantasma\n"
        "Action Input: qualquer"
    )
    resp2 = MagicMock()
    resp2.message.content = (
        "Thought: A ferramenta não existia.\n"
        "Final Answer: Não consegui usar a ferramenta."
    )
    mock_chat.side_effect = [resp1, resp2]

    agent = _make_agent()
    result = agent.run("Teste de ferramenta inexistente")

    assert not result.truncated
    step = result.steps[0]
    assert "não encontrada" in step.observation.lower()


# ── AgentResult e Step — integridade dos dataclasses ────────────────────────


def test_agent_result_defaults():
    """AgentResult deve ter valores padrão corretos."""
    r = AgentResult(answer="ok")
    assert r.steps == []
    assert r.truncated is False
    assert r.total_steps == 0


def test_step_armazena_campos():
    """Step deve armazenar todos os campos corretamente."""
    s = Step(thought="T", action="A", action_input="I", observation="O")
    assert s.thought == "T"
    assert s.action == "A"
    assert s.action_input == "I"
    assert s.observation == "O"
