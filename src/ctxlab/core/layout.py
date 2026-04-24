from typing import Literal

from inspect_ai.model import ChatMessage, ChatMessageSystem, ChatMessageUser

Layout = Literal[
    "baseline",
    "query_first",
    "query_sandwich",
    "q_repeat_end",
    "isolated_sections",
]

# Layout-matched system prompts. Each prompt describes the order the model
# will actually see, to keep the instruction fair across layouts.
SYSTEM_PROMPTS: dict[str, str] = {
    "baseline": (
        "You will receive a long context split across multiple messages. "
        "Use ONLY that context to answer the question at the end. "
        "Respond with just the answer (no extra words)."
    ),
    "query_first": (
        "You will receive a question followed by a long context split across "
        "multiple messages. Use ONLY that context to answer the question. "
        "Respond with just the answer (no extra words)."
    ),
    "query_sandwich": (
        "You will receive a long context split across multiple messages, "
        "with the question appearing at the start and end. "
        "Use ONLY that context to answer the question. "
        "Respond with just the answer (no extra words)."
    ),
    "q_repeat_end": (
        "You will receive a long context split across multiple messages. "
        "Use ONLY that context to answer the question at the end. "
        "Respond with just the answer (no extra words)."
    ),
    "isolated_sections": (
        "You will receive evidence and a question in clearly marked sections. "
        "Use ONLY the evidence to answer the question. "
        "Respond with just the answer (no extra words)."
    ),
}

_FALLBACK_SYSTEM_PROMPT = (
    "You will receive a long context split across multiple messages. "
    "Use ONLY that context to answer the question. "
    "Respond with just the answer (no extra words)."
)


def system_prompt_for_layout(layout: Layout) -> str:
    return SYSTEM_PROMPTS.get(layout, _FALLBACK_SYSTEM_PROMPT)


def apply_layout(
    *,
    layout: Layout,
    system: ChatMessageSystem,
    context_msgs: list[ChatMessageUser],
    question: str,
) -> list[ChatMessage]:
    q = ChatMessageUser(content=question)

    if layout == "baseline":
        return [system, *context_msgs, q]
    if layout == "query_first":
        return [system, q, *context_msgs]
    if layout == "query_sandwich":
        return [system, q, *context_msgs, q]
    if layout == "q_repeat_end":
        return [system, *context_msgs, q, q]
    if layout == "isolated_sections":
        header_ev = ChatMessageUser(content="## EVIDENCE")
        header_q = ChatMessageUser(content="## QUESTION")
        return [system, header_ev, *context_msgs, header_q, q]

    raise ValueError(f"Unknown layout: {layout}")
