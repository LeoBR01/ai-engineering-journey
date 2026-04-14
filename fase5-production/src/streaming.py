"""Formatação de respostas como Server-Sent Events (SSE).

Formato SSE padrão:
  data: <conteúdo>\n\n

Evento de encerramento:
  data: [DONE]\n\n
"""

from collections.abc import Iterator

SSE_DONE = "data: [DONE]\n\n"


def to_sse(tokens: Iterator[str]) -> Iterator[str]:
    """Converte um iterador de tokens em eventos SSE.

    Args:
        tokens: iterador de strings (tokens do LLM).

    Yields:
        Strings no formato SSE, terminando com SSE_DONE.
    """
    for token in tokens:
        yield f"data: {token}\n\n"
    yield SSE_DONE


def string_to_sse(text: str) -> Iterator[str]:
    """Converte uma string completa em um evento SSE único + DONE.

    Usado quando a resposta já está pronta (cache hit, agent).

    Args:
        text: string completa a enviar.

    Yields:
        Evento com o texto completo, seguido de SSE_DONE.
    """
    yield f"data: {text}\n\n"
    yield SSE_DONE
