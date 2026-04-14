from src.streaming import SSE_DONE, string_to_sse, to_sse


def test_to_sse_formats_each_token() -> None:
    events = list(to_sse(iter(["hello", " world"])))
    assert events[0] == "data: hello\n\n"
    assert events[1] == "data:  world\n\n"


def test_to_sse_ends_with_done() -> None:
    events = list(to_sse(iter(["token"])))
    assert events[-1] == SSE_DONE


def test_to_sse_empty_iterator() -> None:
    events = list(to_sse(iter([])))
    assert events == [SSE_DONE]


def test_string_to_sse_produces_two_events() -> None:
    events = list(string_to_sse("full answer here"))
    assert len(events) == 2
    assert events[0] == "data: full answer here\n\n"
    assert events[1] == SSE_DONE


def test_sse_done_constant_format() -> None:
    assert SSE_DONE == "data: [DONE]\n\n"
