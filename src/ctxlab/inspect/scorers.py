from __future__ import annotations

import re
import unicodedata

from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Metric,
    Score,
    Scorer,
    Target,
    mean,
    metric,
    scorer,
)
from inspect_ai.solver import TaskState

_ANSWER_PREFIX_RE = re.compile(
    r"^(?:the\s+answer\s+is|answer\s*:|the\s+correct\s+answer\s+is)\s*",
    re.IGNORECASE,
)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def normalize_answer(text: str) -> str:
    # Strip reasoning tags, answer prefixes, punctuation, articles, and casing,
    # so that reasoning-style models do not fail EM on correct content.
    text = _THINK_BLOCK_RE.sub("", text)
    text = _ANSWER_PREFIX_RE.sub("", text)
    text = unicodedata.normalize("NFKD", text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())
    return text.strip()


@scorer(metrics=[mean()])
def normalized_exact() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        model_answer = state.output.completion if state.output else ""
        pred = normalize_answer(model_answer)
        ref = normalize_answer(target.text)
        return Score(
            value=CORRECT if pred == ref else INCORRECT,
            answer=model_answer,
            explanation=f"normalized: {pred!r} vs {ref!r}",
        )
    return score


@metric
def context_efficiency() -> Metric:
    def metric_fn(scores: list[Score]) -> dict[str, float]:
        total_tb = total_ta = total_mb = total_ma = total_n = 0
        for s in scores:
            if not (s.metadata and "context_stats" in s.metadata):
                continue
            stats_list = s.metadata["context_stats"]
            if not stats_list:
                continue
            last = stats_list[-1]
            total_tb += last.get("tokens_before", 0)
            total_ta += last.get("tokens_after", 0)
            total_mb += last.get("messages_before", 0)
            total_ma += last.get("messages_after", 0)
            total_n += 1

        if total_n == 0:
            return {}

        out = {
            "avg_tokens_before": total_tb / total_n,
            "avg_tokens_after": total_ta / total_n,
            "avg_messages_before": total_mb / total_n,
            "avg_messages_after": total_ma / total_n,
            "token_reduction_pct": (total_tb - total_ta) / total_tb * 100 if total_tb > 0 else 0,
            "message_reduction_pct": (total_mb - total_ma) / total_mb * 100 if total_mb > 0 else 0,
        }

        hits = sum(1 for s in scores if s.metadata and s.metadata.get("needle_in_evidence") is True)
        total = sum(1 for s in scores if s.metadata and s.metadata.get("needle_in_evidence") is not None)
        if total > 0:
            out["needle_selected_rate"] = hits / total
        return out

    return metric_fn


@scorer(metrics=[context_efficiency()])
def context_stats_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        context_stats = state.metadata.get("context_stats", [])

        if context_stats:
            saved = sum(
                s.get("tokens_before", 0) - s.get("tokens_after", 0) for s in context_stats
            )
            filtered = sum(
                s.get("messages_before", 0) - s.get("messages_after", 0) for s in context_stats
            )
            # Score.value gets aggregated numerically by Inspect, so keep strings out of it.
            value: dict = {
                "tokens_saved": float(saved),
                "messages_filtered": float(filtered),
                "filter_calls": float(len(context_stats)),
            }
        else:
            value = {"no_stats": 0.0}

        meta: dict = {
            "context_stats": context_stats,
            "needle_in_evidence": state.metadata.get("needle_in_evidence"),
        }
        if context_stats:
            meta["policy_name"] = context_stats[0].get("policy_name", "unknown")

        return Score(
            value=value,
            explanation=f"Context stats collected: {len(context_stats)} filter operations",
            metadata=meta,
        )

    return score
