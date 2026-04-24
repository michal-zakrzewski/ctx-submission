from __future__ import annotations

from typing import Any, List, Protocol

from inspect_ai.model import ChatMessage, ContentText


class ContextPolicy(Protocol):
    name: str

    def select(self, messages: List[ChatMessage], query: str) -> List[ChatMessage]: ...
    def compress(self, selected: List[ChatMessage], query: str) -> str: ...


def message_text(message: ChatMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    if isinstance(message.content, list):
        parts = []
        for item in message.content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, ContentText):
                parts.append(item.text)
        return "\n".join(parts)
    return str(message.content)


def needle_in_messages(messages: List[Any], needle_text: str) -> bool:
    # Also matches needles that span a chunk boundary.
    texts = [str(getattr(m, "content", m)) for m in messages]
    for t in texts:
        if needle_text in t:
            return True
    for i in range(len(texts) - 1):
        if needle_text in (texts[i] + " " + texts[i + 1]):
            return True
    return False
