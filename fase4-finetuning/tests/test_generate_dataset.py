"""Testes para geração de dataset de fine-tuning."""

from src.generate_dataset import (
    RagPair,
    filter_rag_pair,
    generate_rag_pair,
    pair_to_chatml_rag,
)


class TestRagPairFilter:
    def test_valid_pair_passes(self) -> None:
        pair = RagPair(
            context="Some context text about machine learning.",
            question="What is machine learning?",
            answer=(
                "Machine learning is a method of data analysis"
                " that automates model building."
            ),
        )
        assert filter_rag_pair(pair) is True

    def test_short_answer_fails(self) -> None:
        pair = RagPair(
            context="Context text here.",
            question="What is it?",
            answer="Short.",
        )
        assert filter_rag_pair(pair) is False

    def test_short_question_fails(self) -> None:
        pair = RagPair(
            context="Context text here.",
            question="Why?",
            answer=(
                "Because this is a long enough answer that passes"
                " the minimum length check."
            ),
        )
        assert filter_rag_pair(pair) is False


class TestGenerateRagPair:
    def test_valid_response_parsed_correctly(self) -> None:
        def llm_fn(prompt: str) -> str:
            return (
                "QUESTION: What are transformers used for?\n"
                "ANSWER: Transformers are used for natural language"
                " processing tasks such as translation and summarization."
            )

        chunk = "Transformers are a deep learning architecture used in NLP."
        pair = generate_rag_pair(chunk, llm_fn)

        assert pair is not None
        assert pair.context == chunk
        assert pair.question == "What are transformers used for?"
        assert "Transformers are used for" in pair.answer

    def test_malformed_response_returns_none(self) -> None:
        def llm_fn(prompt: str) -> str:
            return "This is not in the expected format."

        pair = generate_rag_pair("Some chunk text.", llm_fn)
        assert pair is None

    def test_prompt_includes_chunk_text(self) -> None:
        captured_prompts = []

        def llm_fn(prompt: str) -> str:
            captured_prompts.append(prompt)
            return (
                "QUESTION: Test question here long enough?\n"
                "ANSWER: Test answer here that is long enough"
                " to pass the filter check."
            )

        generate_rag_pair("unique_chunk_marker_xyz", llm_fn)
        assert len(captured_prompts) == 1
        assert "unique_chunk_marker_xyz" in captured_prompts[0]


class TestPairToChatml:
    def test_output_has_three_messages(self) -> None:
        pair = RagPair(
            context="Context about RAG.",
            question="What is RAG?",
            answer="RAG is retrieval-augmented generation.",
        )
        result = pair_to_chatml_rag(pair)

        assert "messages" in result
        assert len(result["messages"]) == 3

    def test_roles_are_correct(self) -> None:
        pair = RagPair(context="Context.", question="Question?", answer="Answer.")
        result = pair_to_chatml_rag(pair)
        roles = [m["role"] for m in result["messages"]]
        assert roles == ["system", "user", "assistant"]

    def test_context_in_user_message(self) -> None:
        pair = RagPair(
            context="unique_context_xyz",
            question="Question?",
            answer="Answer here.",
        )
        result = pair_to_chatml_rag(pair)
        user_content = result["messages"][1]["content"]
        assert "unique_context_xyz" in user_content

    def test_answer_in_assistant_message(self) -> None:
        pair = RagPair(
            context="Context.", question="Question?", answer="unique_answer_xyz"
        )
        result = pair_to_chatml_rag(pair)
        assert result["messages"][2]["content"] == "unique_answer_xyz"
