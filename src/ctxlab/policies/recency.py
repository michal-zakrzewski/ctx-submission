from __future__ import annotations

from typing import List

from inspect_ai.model import ChatMessage

from ctxlab.core.types import message_text


class RecencyPolicy:
    """Keep the last k messages regardless of content."""

    name = "recency"

    def __init__(self, k: int = 50):
        self.k = k

    def select(self, messages: List[ChatMessage], query: str) -> List[ChatMessage]:
        return messages[-self.k:]

    def compress(self, selected: List[ChatMessage], query: str) -> str:
        return "\n\n".join(message_text(m) for m in selected)
