from inspect_ai.model import ChatMessageSystem, ChatMessageUser

from ctxlab.core.layout import Layout, apply_layout, system_prompt_for_layout


def _build(layout: Layout):
    sys = ChatMessageSystem(content=system_prompt_for_layout(layout))
    ctx = [ChatMessageUser(content=f"chunk-{i}") for i in range(3)]
    return apply_layout(layout=layout, system=sys, context_msgs=ctx, question="Q?")


def test_baseline_order():
    msgs = _build("baseline")
    assert msgs[0].role == "system"
    assert [str(m.content) for m in msgs[1:-1]] == ["chunk-0", "chunk-1", "chunk-2"]
    assert str(msgs[-1].content) == "Q?"


def test_query_first_places_question_before_evidence():
    msgs = _build("query_first")
    assert str(msgs[1].content) == "Q?"
    assert [str(m.content) for m in msgs[2:]] == ["chunk-0", "chunk-1", "chunk-2"]


def test_query_sandwich_wraps_evidence():
    msgs = _build("query_sandwich")
    assert str(msgs[1].content) == "Q?"
    assert str(msgs[-1].content) == "Q?"
    assert [str(m.content) for m in msgs[2:-1]] == ["chunk-0", "chunk-1", "chunk-2"]


def test_q_repeat_end_duplicates_question():
    msgs = _build("q_repeat_end")
    assert [str(m.content) for m in msgs[-2:]] == ["Q?", "Q?"]


def test_isolated_sections_adds_two_markers():
    msgs = _build("isolated_sections")
    contents = [str(m.content) for m in msgs]
    assert "## EVIDENCE" in contents
    assert "## QUESTION" in contents
    assert contents.index("## EVIDENCE") < contents.index("## QUESTION")
