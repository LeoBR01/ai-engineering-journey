"""Testes do módulo de geração (src/generation.py).

Verifica a construção do prompt, a chamada ao Ollama e o retorno
da resposta. O Ollama é sempre mockado para evitar dependência
de um servidor local nos testes.
"""

from unittest.mock import MagicMock, patch

from src.generation import build_prompt, generate

SAMPLE_CHUNKS = [
    {
        "text": "trecho sobre redes neurais",
        "source": "paper.pdf",
        "page": 2,
        "score": 0.95,
    },  # noqa: E501
    {
        "text": "trecho sobre transformers",
        "source": "paper.pdf",
        "page": 5,
        "score": 0.87,
    },  # noqa: E501
]


def _mock_chat_response(content: str = "Resposta gerada pelo LLM") -> MagicMock:
    """Cria um ChatResponse falso compatível com ollama.chat()."""
    resp = MagicMock()
    resp.message.content = content
    return resp


class TestBuildPrompt:
    def test_inclui_contexto_no_system_message(self):
        messages = build_prompt("meu contexto aqui", "pergunta?")
        system_content = messages[0]["content"]
        assert "meu contexto aqui" in system_content

    def test_inclui_pergunta_na_mensagem_do_usuario(self):
        messages = build_prompt("contexto", "qual é o conceito de atenção?")
        user_content = messages[1]["content"]
        assert "qual é o conceito de atenção?" in user_content

    def test_retorna_lista_com_duas_mensagens(self):
        messages = build_prompt("ctx", "pergunta")
        assert len(messages) == 2

    def test_roles_sao_system_e_user(self):
        messages = build_prompt("ctx", "pergunta")
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"


class TestGenerate:
    def test_retorna_string_nao_vazia(self):
        with patch(
            "src.generation.ollama.chat",
            return_value=_mock_chat_response("Redes neurais são..."),
        ):
            result = generate("O que são redes neurais?", SAMPLE_CHUNKS)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_chama_ollama_com_modelo_correto(self):
        with patch(
            "src.generation.ollama.chat",
            return_value=_mock_chat_response(),
        ) as mock_chat:
            generate("pergunta", SAMPLE_CHUNKS, model="llama3.2")
        assert mock_chat.call_args.kwargs["model"] == "llama3.2"

    def test_passa_contexto_no_prompt(self):
        with patch(
            "src.generation.ollama.chat",
            return_value=_mock_chat_response(),
        ) as mock_chat:
            generate("pergunta", SAMPLE_CHUNKS)
        messages = mock_chat.call_args.kwargs["messages"]
        system_content = messages[0]["content"]
        assert "trecho sobre redes neurais" in system_content
        assert "trecho sobre transformers" in system_content
