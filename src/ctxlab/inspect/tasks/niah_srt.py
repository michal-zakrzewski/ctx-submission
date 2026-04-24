"""Chunked-chat NIAH (needle-in-a-haystack / single-needle retrieval) built from
OpenCompass NeedleBench.

The context is stored as many user messages (one chunk per message) so that a
ContextPolicy can filter and reorder at message granularity, which is the
premise of the evidence-controlled layout protocol.
"""

from __future__ import annotations

import hashlib
import random
from typing import Literal, Sequence

from datasets import load_dataset
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser
from inspect_ai.scorer import exact, f1
from inspect_ai.solver import Solver

from ctxlab.core.layout import Layout, system_prompt_for_layout
from ctxlab.inspect.scorers import context_stats_scorer, normalized_exact
from ctxlab.inspect.solvers import niah_context_managed_solver
from ctxlab.policies.bm25 import BM25Policy
from ctxlab.policies.keyword import KeywordPolicy
from ctxlab.policies.recency import RecencyPolicy

PolicyName = Literal["recency", "keyword", "bm25"]
SampleMethod = Literal["fixed", "sequential", "random"]
Language = Literal["en", "zh", "English", "Chinese"]


def _canonical_language(lang: str) -> str:
    # NeedleBench uses 'English' / 'Chinese' labels, not 'en' / 'zh'.
    key = (lang or "").strip().lower()
    return {
        "en": "English",
        "english": "English",
        "zh": "Chinese",
        "chinese": "Chinese",
        "cn": "Chinese",
        "zh_cn": "Chinese",
    }.get(key, lang)


def _haystack_config_for_language(canon_language: str) -> str:
    return "en_haystack_texts" if canon_language == "English" else "zh_haystack_texts"


def _make_policy(policy_name: PolicyName, k: int):
    if policy_name == "recency":
        return RecencyPolicy(k=k)
    if policy_name == "keyword":
        return KeywordPolicy(k=k)
    if policy_name == "bm25":
        return BM25Policy(k=k)
    raise ValueError(f"Unknown policy: {policy_name}")


def _linspace_int(a: int, b: int, n: int) -> list[int]:
    if n <= 1:
        return [int(a)]
    step = (b - a) / (n - 1)
    return [int(round(a + i * step)) for i in range(n)]


def _get_text(rec: dict) -> str:
    for key in ("text", "context", "content", "document"):
        v = rec.get(key)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def _stable_seed(*parts: object) -> int:
    # Deterministic across Python invocations (unlike hash()).
    s = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(s, digest_size=8).digest(), "big", signed=False)


def _depth_pcts(
    context_words: int,
    n_positions: int,
    start_buffer_words: int,
    end_buffer_words: int,
) -> list[float]:
    if n_positions <= 1:
        return [50.0]
    context_words = max(1, context_words)
    min_pct = max(0.0, min(100.0, 100.0 * start_buffer_words / context_words))
    max_pct = max(0.0, min(100.0, 100.0 * (context_words - end_buffer_words) / context_words))
    if max_pct < min_pct:
        max_pct = min_pct
    return [min_pct + (max_pct - min_pct) * i / (n_positions - 1) for i in range(n_positions)]


def _pick_needle_conditioned(
    needles: Sequence[dict],
    method: SampleMethod,
    fixed_index: int,
    *,
    seed: int,
    run_idx: int,
    context_words: int,
    depth_pct: float,
    rep: int,
) -> tuple[dict, int]:
    # Seed needle choice from the condition itself so selection is
    # decorrelated from the nested loop order over (context_words, depth_pct).
    n = len(needles)
    if n == 0:
        raise ValueError("No needles available after filtering.")

    if method == "fixed":
        idx = fixed_index % n
        return needles[idx], idx

    cond_seed = _stable_seed(seed, run_idx, context_words, f"{depth_pct:.6f}", rep)
    r = random.Random(cond_seed)

    if method == "random":
        idx = r.randrange(n)
    elif method == "sequential":
        idx = (fixed_index + r.randrange(n)) % n
    else:
        raise ValueError(f"Unknown sample_method: {method}")

    return needles[idx], idx


def _build_haystack_words(haystacks, rng: random.Random, n_words: int) -> list[str]:
    words: list[str] = []
    n_words = max(1, n_words)
    while len(words) < n_words:
        rec = haystacks[rng.randrange(len(haystacks))]
        txt = _get_text(rec)
        if not txt:
            continue
        words.extend(txt.split())
    return words[:n_words]


def _chunk_messages(words: list[str], chunk_words: int) -> list[ChatMessageUser]:
    chunk_words = max(1, chunk_words)
    msgs: list[ChatMessageUser] = []
    for i in range(0, len(words), chunk_words):
        chunk = " ".join(words[i : i + chunk_words])
        msgs.append(ChatMessageUser(content=chunk))
    return msgs


def _find_needle_chunk(context_msgs: list[ChatMessageUser], needle_text: str) -> int | None:
    for i, msg in enumerate(context_msgs):
        if needle_text in str(msg.content):
            return i
    # Also detect needles that span a chunk boundary.
    for i in range(len(context_msgs) - 1):
        combined = str(context_msgs[i].content) + " " + str(context_msgs[i + 1].content)
        if needle_text in combined:
            return i
    return None


def _make_solver(
    policy_name: PolicyName,
    policy_k: int,
    token_budget: int,
    preserve_system: bool,
    preserve_question: bool,
    layout: Layout,
) -> Solver:
    policy = _make_policy(policy_name, policy_k)
    return niah_context_managed_solver(
        policy=policy,
        token_budget=token_budget,
        preserve_system=preserve_system,
        preserve_question=preserve_question,
        layout=layout,
    )


@task
def niah_srt_with_context_policy(
    language: Language | str = "en",
    seed: int = 1234,
    min_context_words: int = 2000,
    max_context_words: int = 8000,
    n_contexts: int = 3,
    n_positions: int = 5,
    start_buffer_words: int = 0,
    end_buffer_words: int = 0,
    n_needles: int = 1,
    n_runs: int = 1,
    sample_method: SampleMethod = "fixed",
    fixed_index: int = 0,
    chunk_words: int = 200,
    policy_name: PolicyName = "recency",
    policy_k: int = 50,
    token_budget: int = 8000,
    preserve_system: bool = True,
    preserve_question: bool = True,
    layout: Layout = "query_first",
) -> Task:
    """Build an Inspect Task for chunked NIAH / single-needle retrieval.

    Pipeline per sample:
      1. Sample haystack text from NeedleBench subsets until we hit ``context_words``.
      2. Insert one needle string at the depth given by ``depth_pct``.
      3. Split into ``ceil(context_words / chunk_words)`` user messages.
      4. Append the retrieval question as the final user message.

    Context lengths and chunk sizes are in whitespace-separated words, not tokens.
    """
    canon_language = _canonical_language(str(language))

    needles_ds = load_dataset("opencompass/NeedleBench", name="retrieval_needles", split="test")
    haystack_cfg = _haystack_config_for_language(canon_language)
    haystacks_ds = load_dataset("opencompass/NeedleBench", name=haystack_cfg, split="test")

    needles = [r for r in needles_ds if r.get("language") == canon_language]
    if not needles:
        available = sorted({r.get("language") for r in needles_ds if r.get("language")})
        raise ValueError(
            f"No needles found for language={canon_language!r}. "
            f"Available languages: {available}."
        )

    context_lengths = _linspace_int(min_context_words, max_context_words, n_contexts)
    system = ChatMessageSystem(content=system_prompt_for_layout(layout))
    samples: list[Sample] = []

    for run_idx in range(max(1, n_runs)):
        rng = random.Random(seed + run_idx)

        for context_words in context_lengths:
            depths = _depth_pcts(context_words, n_positions, start_buffer_words, end_buffer_words)
            for depth_pct in depths:
                for rep in range(n_needles):
                    needle_rec, needle_idx = _pick_needle_conditioned(
                        needles=needles,
                        method=sample_method,
                        fixed_index=fixed_index,
                        seed=seed,
                        run_idx=run_idx,
                        context_words=context_words,
                        depth_pct=depth_pct,
                        rep=rep,
                    )

                    needle_text = str(needle_rec.get("needle", "")).strip()
                    question = str(needle_rec.get("retrieval_question", "")).strip()
                    target = str(needle_rec.get("gold_standard_answer", "")).strip()
                    if not needle_text or not question or not target:
                        continue

                    needle_words = needle_text.split()
                    base_len = max(1, context_words - max(1, len(needle_words)))
                    hay_words = _build_haystack_words(haystacks_ds, rng=rng, n_words=base_len)

                    start_buf = min(start_buffer_words, base_len)
                    end_buf = min(end_buffer_words, base_len)
                    lo = start_buf
                    hi = max(lo, base_len - end_buf)
                    idx = int(round((depth_pct / 100.0) * base_len))
                    idx = max(lo, min(hi, idx))

                    full_words = hay_words[:idx] + needle_words + hay_words[idx:]
                    full_words = full_words[:context_words]

                    context_msgs = _chunk_messages(full_words, chunk_words=chunk_words)
                    question_msg = ChatMessageUser(content=question)
                    messages = [system, *context_msgs, question_msg]
                    needle_chunk_idx = _find_needle_chunk(context_msgs, needle_text)

                    d = int(round(depth_pct))
                    sample_id = (
                        f"niah-{canon_language}-{run_idx}-L{context_words}-D{d}"
                        f"-N{needle_idx}-R{rep}-CL{layout}"
                    )

                    samples.append(
                        Sample(
                            id=sample_id,
                            input=messages,
                            target=target,
                            metadata={
                                "run_idx": run_idx,
                                "language": canon_language,
                                "haystack_cfg": haystack_cfg,
                                "context_words": context_words,
                                "depth_pct": float(depth_pct),
                                "needle_index": int(needle_idx),
                                "needle_len_words": len(needle_words),
                                "chunk_words": chunk_words,
                                "sample_method": sample_method,
                                "layout": layout,
                                "needle_text": needle_text,
                                "needle_chunk_idx": needle_chunk_idx,
                            },
                        )
                    )

    solver = _make_solver(
        policy_name=policy_name,
        policy_k=policy_k,
        token_budget=token_budget,
        preserve_system=preserve_system,
        preserve_question=preserve_question,
        layout=layout,
    )

    return Task(
        dataset=MemoryDataset(samples, name="niah_opencompass_chunked"),
        solver=solver,
        scorer=[f1(), exact(), normalized_exact(), context_stats_scorer()],
    )
