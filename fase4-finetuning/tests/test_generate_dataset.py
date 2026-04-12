"""Testes para geração de dataset de fine-tuning."""

import json
import tempfile
from pathlib import Path

from src.generate_dataset import (
    RagPair,
    filter_rag_pair,
    format_observation,
    generate_rag_pair,
    generate_react_cycle,
    pair_to_chatml_rag,
    preprocess_dataset,
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


class TestFormatObservation:
    def test_formats_top_3_results(self) -> None:
        results = [
            {"content": "Result one about transformers."},
            {"content": "Result two about attention."},
            {"content": "Result three about BERT."},
            {"content": "Result four should be excluded."},
        ]
        observation = format_observation(results)
        assert "[1]" in observation
        assert "[2]" in observation
        assert "[3]" in observation
        assert "[4]" not in observation

    def test_empty_results_returns_fallback(self) -> None:
        observation = format_observation([])
        assert "Nenhum resultado encontrado" in observation

    def test_content_truncated_to_300_chars(self) -> None:
        long_content = "x" * 500
        results = [{"content": long_content}]
        observation = format_observation(results)
        # 300 chars + prefix "[1] "
        assert len(observation) <= 310


class TestGenerateReactCycle:
    def test_cycle_has_correct_chatml_structure(self) -> None:
        def search_fn(query: str) -> list[dict]:
            return [{"content": "Relevant paper content about the topic."}]

        def generate_fn(question: str, context: str) -> str:
            return "The answer based on the retrieved context."

        result = generate_react_cycle(
            question="What is attention mechanism?",
            search_fn=search_fn,
            generate_fn=generate_fn,
            system_prompt="You are a ReAct agent.",
        )

        assert "messages" in result
        assert len(result["messages"]) == 3
        roles = [m["role"] for m in result["messages"]]
        assert roles == ["system", "user", "assistant"]

    def test_assistant_message_has_react_keywords(self) -> None:
        def search_fn(query: str) -> list[dict]:
            return [{"content": "Paper content."}]

        def generate_fn(question: str, context: str) -> str:
            return "Final answer here."

        result = generate_react_cycle(
            question="How does RAG work?",
            search_fn=search_fn,
            generate_fn=generate_fn,
            system_prompt="System prompt.",
        )
        assistant_content = result["messages"][2]["content"]

        assert "Thought:" in assistant_content
        assert "Action: search_papers" in assistant_content
        assert "Action Input:" in assistant_content
        assert "Observation:" in assistant_content
        assert "Final Answer:" in assistant_content

    def test_search_fn_called_with_question(self) -> None:
        captured_queries = []

        def search_fn(query: str) -> list[dict]:
            captured_queries.append(query)
            return [{"content": "Some content."}]

        def generate_fn(question: str, context: str) -> str:
            return "Answer."

        generate_react_cycle(
            question="unique_question_marker",
            search_fn=search_fn,
            generate_fn=generate_fn,
            system_prompt="Prompt.",
        )

        assert len(captured_queries) == 1
        assert "unique_question_marker" in captured_queries[0]

    def test_generate_fn_called_with_observation(self) -> None:
        captured_contexts = []

        def search_fn(query: str) -> list[dict]:
            return [{"content": "unique_content_xyz"}]

        def generate_fn(question: str, context: str) -> str:
            captured_contexts.append(context)
            return "Answer."

        generate_react_cycle(
            question="Question?",
            search_fn=search_fn,
            generate_fn=generate_fn,
            system_prompt="Prompt.",
        )

        assert len(captured_contexts) == 1
        assert "unique_content_xyz" in captured_contexts[0]

    def test_system_prompt_in_first_message(self) -> None:
        def search_fn(query: str) -> list[dict]:
            return []

        def generate_fn(question: str, context: str) -> str:
            return "Answer."

        result = generate_react_cycle(
            question="Q?",
            search_fn=search_fn,
            generate_fn=generate_fn,
            system_prompt="unique_system_prompt_xyz",
        )

        assert result["messages"][0]["content"] == "unique_system_prompt_xyz"


class TestPreprocessDataset:
    def _write_jsonl(self, path: Path, items: list[dict]) -> None:
        with open(path, "w") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")

    def test_output_files_created(self) -> None:
        rag_item = {
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ]
        }
        react_item = {
            "messages": [
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": "a2"},
            ]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            rag_path = Path(tmpdir) / "rag.jsonl"
            react_path = Path(tmpdir) / "react.jsonl"
            output_path = Path(tmpdir) / "processed" / "train.jsonl"

            self._write_jsonl(rag_path, [rag_item] * 10)
            self._write_jsonl(react_path, [react_item] * 10)

            result = preprocess_dataset(
                str(rag_path), str(react_path), str(output_path)
            )

            assert output_path.exists()
            assert (output_path.parent / "val.jsonl").exists()
            assert "train" in result
            assert "val" in result

    def test_total_count_preserved(self) -> None:
        rag_item = {
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ]
        }
        react_item = {
            "messages": [
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": "a2"},
            ]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            rag_path = Path(tmpdir) / "rag.jsonl"
            react_path = Path(tmpdir) / "react.jsonl"
            output_path = Path(tmpdir) / "processed" / "train.jsonl"

            self._write_jsonl(rag_path, [rag_item] * 100)
            self._write_jsonl(react_path, [react_item] * 100)

            result = preprocess_dataset(
                str(rag_path), str(react_path), str(output_path)
            )

            assert result["train"] + result["val"] == 200

    def test_val_ratio_respected(self) -> None:
        rag_item = {
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ]
        }
        react_item = {
            "messages": [
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": "a2"},
            ]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            rag_path = Path(tmpdir) / "rag.jsonl"
            react_path = Path(tmpdir) / "react.jsonl"
            output_path = Path(tmpdir) / "processed" / "train.jsonl"

            self._write_jsonl(rag_path, [rag_item] * 50)
            self._write_jsonl(react_path, [react_item] * 50)

            result = preprocess_dataset(
                str(rag_path), str(react_path), str(output_path), val_ratio=0.1
            )

            assert result["val"] == 10  # 10% of 100

    def test_reproducible_with_same_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            rag_path = Path(tmpdir) / "rag.jsonl"
            react_path = Path(tmpdir) / "react.jsonl"

            self._write_jsonl(
                rag_path,
                [
                    {"messages": [{"role": "user", "content": str(i)}]}
                    for i in range(30)
                ],
            )
            self._write_jsonl(
                react_path,
                [
                    {"messages": [{"role": "user", "content": str(i + 100)}]}
                    for i in range(30)
                ],
            )

            output1 = Path(tmpdir) / "out1" / "train.jsonl"
            output2 = Path(tmpdir) / "out2" / "train.jsonl"

            preprocess_dataset(str(rag_path), str(react_path), str(output1), seed=42)
            preprocess_dataset(str(rag_path), str(react_path), str(output2), seed=42)

            assert output1.read_text() == output2.read_text()
