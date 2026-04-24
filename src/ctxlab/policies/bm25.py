from __future__ import annotations

import math
from typing import Dict, List

from inspect_ai.model import ChatMessage

from ctxlab.core.types import message_text


class BM25Policy:
    """Standard BM25 over message-granularity chunks. Falls back to recency on empty query."""

    name = "bm25"

    def __init__(self, k: int = 50, k1: float = 1.5, b: float = 0.75):
        self.k = k
        self.k1 = k1
        self.b = b

    def select(self, messages: List[ChatMessage], query: str) -> List[ChatMessage]:
        if not query.strip():
            return messages[-self.k:]

        query_terms = query.lower().split()
        if not query_terms:
            return messages[-self.k:]

        tokenised: List[List[str]] = [message_text(m).lower().split() for m in messages]

        doc_freq: Dict[str, int] = {}
        for tokens in tokenised:
            for t in set(tokens):
                doc_freq[t] = doc_freq.get(t, 0) + 1

        N = len(messages)
        idf: Dict[str, float] = {}
        for t in query_terms:
            df = doc_freq.get(t, 0)
            idf[t] = math.log((N - df + 0.5) / (df + 0.5) + 1.0)

        total_len = sum(len(tokens) for tokens in tokenised)
        avgdl = total_len / N if N > 0 else 0

        scored = []
        for idx, (m, tokens) in enumerate(zip(messages, tokenised)):
            tf: Dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1

            dl = len(tokens)
            s = 0.0
            for t in query_terms:
                if t in tf:
                    num = tf[t] * (self.k1 + 1)
                    den = tf[t] + self.k1 * (1 - self.b + self.b * (dl / avgdl if avgdl > 0 else 1))
                    s += idf[t] * (num / den)
            scored.append((s, idx, m))

        # Sort by score desc; earlier index wins ties.
        scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)
        return [m for _, _, m in scored[: self.k]]

    def compress(self, selected: List[ChatMessage], query: str) -> str:
        return "\n\n".join(message_text(m) for m in selected)
