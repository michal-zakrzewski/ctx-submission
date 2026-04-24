from inspect_ai.model import ChatMessageUser

from ctxlab.policies.bm25 import BM25Policy


def test_bm25_falls_back_to_recency_on_empty_query():
    msgs = [ChatMessageUser(content=f"chunk {i}") for i in range(10)]
    selected = BM25Policy(k=3).select(msgs, "")
    assert [str(m.content) for m in selected] == ["chunk 7", "chunk 8", "chunk 9"]


def test_bm25_prefers_messages_matching_query_terms():
    msgs = [
        ChatMessageUser(content="lorem ipsum"),
        ChatMessageUser(content="the tower of pisa leans"),
        ChatMessageUser(content="a cat sat on a mat"),
    ]
    selected = BM25Policy(k=1).select(msgs, "tower pisa")
    assert str(selected[0].content) == "the tower of pisa leans"
