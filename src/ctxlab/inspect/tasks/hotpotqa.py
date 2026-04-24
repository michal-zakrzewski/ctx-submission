"""HotpotQA (distractor) adapter for the evidence-controlled layout protocol.

Each of the 10 context paragraphs becomes one user message, mirroring the
NIAH-SRT message-granularity setup so the same solvers can be reused.
"""

from __future__ import annotations

from typing import Literal

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

_POLICIES = {
    "recency": RecencyPolicy,
    "keyword": KeywordPolicy,
    "bm25": BM25Policy,
}


def _make_policy(policy_name: PolicyName, k: int):
    cls = _POLICIES.get(policy_name)
    if cls is None:
        raise ValueError(f"Unknown policy: {policy_name}")
    return cls(k=k)


def _make_solver(
    policy_name: PolicyName,
    policy_k: int,
    token_budget: int,
    layout: Layout,
) -> Solver:
    return niah_context_managed_solver(
        policy=_make_policy(policy_name, policy_k),
        token_budget=token_budget,
        preserve_system=True,
        preserve_question=True,
        layout=layout,
    )


@task
def hotpotqa_with_context_policy(
    split: str = "validation",
    limit: int | None = None,
    policy_name: PolicyName = "bm25",
    policy_k: int = 10,
    token_budget: int = 4000,
    layout: Layout = "baseline",
) -> Task:
    """Build an Inspect Task for HotpotQA distractor with layout control."""
    ds = load_dataset("hotpot_qa", "distractor", split=split)
    system = ChatMessageSystem(content=system_prompt_for_layout(layout))

    samples: list[Sample] = []
    for idx, row in enumerate(ds):
        if limit is not None and idx >= limit:
            break

        question = row["question"]
        answer = row["answer"]
        level = row.get("level", "")
        qtype = row.get("type", "")

        titles: list[str] = row["context"]["title"]
        sentences: list[list[str]] = row["context"]["sentences"]
        supporting_titles: set[str] = set(row["supporting_facts"]["title"])

        context_msgs: list[ChatMessageUser] = []
        gold_indices: list[int] = []
        for i, (title, sents) in enumerate(zip(titles, sentences)):
            para = f"[{title}] " + " ".join(sents)
            context_msgs.append(ChatMessageUser(content=para))
            if title in supporting_titles:
                gold_indices.append(i)

        question_msg = ChatMessageUser(content=question)
        messages = [system, *context_msgs, question_msg]

        samples.append(
            Sample(
                id=f"hotpotqa-{split}-{idx}-{layout}",
                input=messages,
                target=answer,
                metadata={
                    "level": level,
                    "type": qtype,
                    "layout": layout,
                    "n_paragraphs": len(context_msgs),
                    "n_gold": len(gold_indices),
                    "gold_indices": gold_indices,
                    "supporting_titles": list(supporting_titles),
                },
            )
        )

    solver = _make_solver(
        policy_name=policy_name,
        policy_k=policy_k,
        token_budget=token_budget,
        layout=layout,
    )

    return Task(
        dataset=MemoryDataset(samples, name="hotpotqa_distractor"),
        solver=solver,
        scorer=[f1(), exact(), normalized_exact(), context_stats_scorer()],
    )
