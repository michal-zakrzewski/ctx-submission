from __future__ import annotations

from inspect_ai.model import ChatMessageSystem, ChatMessageUser
from inspect_ai.solver import Generate, Solver, TaskState, generate, solver

from ctxlab.core.inspect_context_manager import InspectContextManager
from ctxlab.core.layout import Layout, apply_layout
from ctxlab.core.types import ContextPolicy, needle_in_messages


def chain_solvers(*solvers: Solver) -> Solver:
    @solver
    def chain() -> Solver:
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            for step in solvers:
                state = await step(state, generate)
            return state
        return solve
    return chain()


def evidence_selector(
    *,
    policy: ContextPolicy,
    token_budget: int,
    preserve_system: bool,
    preserve_question: bool,
) -> Solver:
    """Stage 1 of the evidence-controlled protocol: pick E' and stash it in state.store."""
    preserve_recent = 1 if preserve_question else 0
    cm = InspectContextManager(
        policy=policy,
        token_budget=token_budget,
        preserve_system=preserve_system,
        preserve_recent=preserve_recent,
    )

    @solver
    def select_evidence() -> Solver:
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            system_msg = next((m for m in state.messages if m.role == "system"), None)
            question_msg = next((m for m in reversed(state.messages) if m.role == "user"), None)

            if system_msg is not None:
                state.store.set("system", system_msg)
            if question_msg is not None:
                state.store.set("question", question_msg)

            pool = list(state.messages)
            if not preserve_question and question_msg is not None:
                pool = [m for m in pool if m is not question_msg]

            query = str(question_msg.content) if question_msg is not None else "current task and context"
            filtered, stats = cm.apply_policy(messages=pool, query=query)

            state.metadata.setdefault("context_stats", []).append(stats)
            state.metadata["full_history_length"] = len(state.messages)

            evidence = [m for m in filtered if m.role != "system" and m is not question_msg]
            state.store.set("evidence", evidence)

            needle_text = state.metadata.get("needle_text")
            if needle_text is not None:
                found = needle_in_messages(evidence, needle_text)
                state.store.set("needle_in_evidence", found)
                state.metadata["needle_in_evidence"] = found

            return state
        return solve
    return select_evidence()


def layout_solver(*, layout: Layout) -> Solver:
    """Stage 2: build the final prompt from stashed evidence under the chosen layout."""
    @solver
    def apply_selected_layout() -> Solver:
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            evidence = state.store.get("evidence", [])
            system_msg = state.store.get("system")
            question_msg = state.store.get("question")

            if system_msg is None:
                system_msg = next((m for m in state.messages if m.role == "system"), None)
            if question_msg is None:
                question_msg = next((m for m in reversed(state.messages) if m.role == "user"), None)

            question_text = str(question_msg.content) if question_msg is not None else ""
            context_msgs = [m for m in evidence if isinstance(m, ChatMessageUser)]

            if system_msg is None:
                system_msg = ChatMessageSystem(content="")

            state.messages = apply_layout(
                layout=layout,
                system=system_msg,
                context_msgs=context_msgs,
                question=question_text,
            )
            return state
        return solve
    return apply_selected_layout()


def niah_context_managed_solver(
    *,
    policy: ContextPolicy,
    token_budget: int,
    preserve_system: bool,
    preserve_question: bool,
    layout: Layout,
) -> Solver:
    ev = evidence_selector(
        policy=policy,
        token_budget=token_budget,
        preserve_system=preserve_system,
        preserve_question=preserve_question,
    )
    lo = layout_solver(layout=layout)
    return chain_solvers(ev, lo, generate())
