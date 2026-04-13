import pytest

from src.router import route


@pytest.mark.parametrize(
    "question,expected",
    [
        ("What is RAG?", "rag"),
        ("How does attention work?", "rag"),
        ("Explain transformers", "rag"),
        ("", "rag"),
        ("Compare RAG and fine-tuning", "agent"),
        ("RAG vs fine-tuning: which is better?", "agent"),
        ("What is the difference between RAG and RLHF?", "agent"),
        ("How does RAG compare to fine-tuning in practice?", "agent"),
        ("Explain both RAG and fine-tuning and their trade-offs", "agent"),
        ("What is the relationship between attention and transformers?", "agent"),
    ],
)
def test_route(question: str, expected: str) -> None:
    assert route(question) == expected
