from __future__ import annotations

from typing import List

from inspect_ai.model import ChatMessage

from .types import ContextPolicy, message_text


class InspectContextManager:
    """Applies a ContextPolicy to an Inspect AI message history and enforces a word budget."""

    def __init__(
        self,
        policy: ContextPolicy,
        token_budget: int = 8000,
        preserve_system: bool = True,
        preserve_recent: int = 3,
    ):
        self.policy = policy
        self.token_budget = token_budget
        self.preserve_system = preserve_system
        self.preserve_recent = preserve_recent

    def apply_policy(
        self,
        messages: List[ChatMessage],
        query: str,
    ) -> tuple[List[ChatMessage], dict]:
        if not messages:
            return [], {
                "messages_before": 0,
                "messages_after": 0,
                "tokens_before": 0,
                "tokens_after": 0,
                "policy_name": self.policy.name,
            }

        messages_before = len(messages)
        tokens_before = sum(len(message_text(m).split()) for m in messages)

        system_messages: list[ChatMessage] = []
        other_messages: list[ChatMessage] = []
        if self.preserve_system:
            for m in messages:
                (system_messages if m.role == "system" else other_messages).append(m)
        else:
            other_messages = list(messages)

        if self.preserve_recent > 0 and len(other_messages) > self.preserve_recent:
            recent = other_messages[-self.preserve_recent:]
            selectable = other_messages[:-self.preserve_recent]
        else:
            recent = []
            selectable = other_messages

        selected = self.policy.select(selectable, query) if selectable else []

        filtered = system_messages + selected + recent
        filtered = self._enforce_budget(filtered)

        return filtered, {
            "messages_before": messages_before,
            "messages_after": len(filtered),
            "tokens_before": tokens_before,
            "tokens_after": sum(len(message_text(m).split()) for m in filtered),
            "policy_name": self.policy.name,
            "system_preserved": len(system_messages),
            "recent_preserved": len(recent),
            "policy_selected": len(selected),
        }

    def _enforce_budget(self, messages: List[ChatMessage]) -> List[ChatMessage]:
        total = sum(len(message_text(m).split()) for m in messages)
        if total <= self.token_budget:
            return messages

        # Drop from the middle; keep first and last windows intact.
        result = list(messages)
        lo = min(3, len(result) // 3)
        hi = max(len(result) - 3, 2 * len(result) // 3)

        while total > self.token_budget and lo < hi:
            dropped = result.pop(lo)
            total -= len(message_text(dropped).split())
            hi -= 1

        return result
