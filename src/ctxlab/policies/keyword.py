from __future__ import annotations

from typing import List

from inspect_ai.model import ChatMessage

from ctxlab.core.types import message_text


class KeywordPolicy:
    """Rank messages by unigram overlap with the query; ties broken by position (later wins)."""

    name = "keyword"

    def __init__(self, k: int = 50):
        self.k = k

    def select(self, messages: List[ChatMessage], query: str) -> List[ChatMessage]:
        query_words = set(query.lower().split())
        scored = []
        for idx, m in enumerate(messages):
            tokens = message_text(m).lower().split()
            score = sum(1 for w in tokens if w in query_words)
            scored.append((score, idx, m))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [m for _, _, m in scored[: self.k]]

    def compress(self, selected: List[ChatMessage], query: str) -> str:
        return "\n\n".join(message_text(m) for m in selected)
