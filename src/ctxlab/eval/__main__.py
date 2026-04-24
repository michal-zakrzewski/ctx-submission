"""CLI for running evidence-controlled layout evaluations through Inspect AI.

Sweeps (policy x budget x layout) cells for a single model. Cells that already
have a success log in ``--log-dir`` are skipped, so long sweeps can be resumed
after an interruption.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, get_args

from inspect_ai import eval as inspect_eval
from inspect_ai.log import EvalLog, list_eval_logs, read_eval_log

from ctxlab.core.layout import Layout
from ctxlab.inspect.tasks.hotpotqa import hotpotqa_with_context_policy
from ctxlab.inspect.tasks.niah_srt import niah_srt_with_context_policy


CellKey = tuple[str, str, int, int, str | None]

_ALLOWED_POLICIES = {"recency", "keyword", "bm25"}
_ALLOWED_BENCHMARKS = ("niah-srt", "hotpotqa")


def _eval_log_issue(log: EvalLog) -> str:
    parts = [f"status={log.status!r}"]
    err = getattr(log, "error", None)
    if err is not None:
        msg = getattr(err, "message", None)
        if msg:
            parts.append(f"message={msg!r}")
    return ", ".join(parts)


def _cell_key(
    model: str, policy_name: str, policy_k: int, token_budget: int, layout: str | None,
) -> CellKey:
    return (model, policy_name, int(policy_k), int(token_budget), layout)


def load_completed_cells(log_dir: str) -> set[CellKey]:
    """Return cells already completed with status=success. Header-only read keeps this cheap."""
    if not Path(log_dir).exists():
        return set()

    completed: set[CellKey] = set()
    for info in list_eval_logs(log_dir):
        try:
            log = read_eval_log(info, header_only=True)
        except Exception:
            continue
        if getattr(log, "status", None) != "success":
            continue

        task_args = dict(getattr(log.eval, "task_args", {}) or {})
        model = getattr(log.eval, "model", None)
        policy_name = task_args.get("policy_name")
        policy_k = task_args.get("policy_k")
        token_budget = task_args.get("token_budget")
        layout = task_args.get("layout")

        if model is None or policy_name is None or policy_k is None or token_budget is None:
            continue

        completed.add(_cell_key(model, policy_name, policy_k, token_budget, layout))

    return completed


def parse_policies(items: str) -> list[dict[str, Any]]:
    """Parse 'recency@50,bm25@30' into a list of {name, k} dicts."""
    policies = []
    for raw in items.split(","):
        raw = raw.strip()
        if not raw:
            continue
        if "@" in raw:
            name, k_str = raw.split("@", 1)
            k = int(k_str)
        else:
            name, k = raw, 50
        name = name.lower()
        if name not in _ALLOWED_POLICIES:
            raise ValueError(f"Unknown policy: {name}")
        policies.append({"name": name, "k": k})
    return policies


def parse_budgets(items: str) -> list[int]:
    return [int(x.strip()) for x in items.split(",") if x.strip()]


def parse_layouts(items: str) -> list[str]:
    layouts = []
    for raw in items.split(","):
        layout = raw.strip()
        if not layout:
            continue
        if layout not in get_args(Layout):
            raise ValueError(f"Unknown layout: {layout}")
        layouts.append(layout)
    return layouts


def run_evaluations(
    policies: list[dict[str, Any]],
    budgets: list[int],
    model: str,
    limit: int | None,
    log_dir: str,
    benchmark: str,
    layouts: list[str],
    niah_language: str,
    niah_min_context_words: int,
    niah_max_context_words: int,
    niah_n_contexts: int,
    niah_n_positions: int,
    niah_start_buffer_words: int,
    niah_end_buffer_words: int,
    niah_n_needles: int,
    niah_n_runs: int,
    niah_sample_method: str,
    niah_fixed_index: int,
    niah_chunk_words: int,
    niah_preserve_system: bool,
    niah_preserve_question: bool,
    message_limit: int | None,
    resume: bool,
) -> list[EvalLog]:
    eval_logs: list[EvalLog] = []

    total_runs = len(policies) * len(budgets) * len(layouts)
    current_run = 0

    completed_cells: set[CellKey] = set()
    if resume:
        completed_cells = load_completed_cells(log_dir)
        if completed_cells:
            print(
                f"[resume] {len(completed_cells)} cell(s) already completed in {log_dir}; "
                "matching combinations will be skipped."
            )

    for policy in policies:
        for budget in budgets:
            for layout in layouts:
                current_run += 1
                tag = f"{policy['name']}@{policy['k']} budget={budget} layout={layout}"

                cell = _cell_key(model, policy["name"], policy["k"], budget, layout)
                if resume and cell in completed_cells:
                    print(f"\n[{current_run}/{total_runs}] skip (already done): {tag}")
                    continue

                print(f"\n[{current_run}/{total_runs}] Running {tag}...")

                if benchmark == "niah-srt":
                    task = niah_srt_with_context_policy(
                        language=niah_language,
                        min_context_words=niah_min_context_words,
                        max_context_words=niah_max_context_words,
                        n_contexts=niah_n_contexts,
                        n_positions=niah_n_positions,
                        start_buffer_words=niah_start_buffer_words,
                        end_buffer_words=niah_end_buffer_words,
                        n_needles=niah_n_needles,
                        n_runs=niah_n_runs,
                        sample_method=niah_sample_method,
                        fixed_index=niah_fixed_index,
                        chunk_words=niah_chunk_words,
                        policy_name=policy["name"],
                        policy_k=policy["k"],
                        token_budget=budget,
                        preserve_system=niah_preserve_system,
                        preserve_question=niah_preserve_question,
                        layout=layout,
                    )
                elif benchmark == "hotpotqa":
                    task = hotpotqa_with_context_policy(
                        policy_name=policy["name"],
                        policy_k=policy["k"],
                        token_budget=budget,
                        layout=layout,
                        limit=limit,
                    )
                else:
                    raise ValueError(f"Unknown benchmark: {benchmark}")

                logs = inspect_eval(
                    task,
                    model=model,
                    log_dir=log_dir,
                    limit=limit,
                    message_limit=message_limit,
                )
                log = logs[0] if logs else None
                if log is None:
                    continue

                eval_logs.append(log)
                if log.status != "success":
                    print(f"  FAIL: {_eval_log_issue(log)}", file=sys.stderr)
                elif getattr(log, "results", None):
                    print(f"  done: {log.results.scores}")

    return eval_logs


def aggregate_results(logs: list[EvalLog], output_path: str | None) -> dict:
    results = []
    for log in logs:
        if not (hasattr(log, "results") and log.results):
            continue

        result = {
            "policy": log.eval.task_args.get("policy_name", "unknown"),
            "policy_k": log.eval.task_args.get("policy_k", 0),
            "token_budget": log.eval.task_args.get("token_budget", 0),
            "layout": log.eval.task_args.get("layout", "unknown"),
            "model": log.eval.model,
        }
        if log.results.scores:
            for score in log.results.scores:
                if hasattr(score, "metrics") and score.metrics:
                    for metric_name, metric in score.metrics.items():
                        result[f"{score.name}_{metric_name}"] = getattr(metric, "value", None)
                elif hasattr(score, "value"):
                    result[score.name] = score.value
        if hasattr(log.results, "metadata") and log.results.metadata:
            if "context_stats" in log.results.metadata:
                result["context_stats"] = log.results.metadata["context_stats"]
        results.append(result)

    aggregated = {"total_runs": len(results), "results": results}

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(aggregated, indent=2))
        print(f"\nWrote aggregated results to {output_path}")

    return aggregated


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run evidence-controlled layout evaluations.")
    p.add_argument("--model", required=True, help="Model id (e.g. ollama/llama3.2:3b)")
    p.add_argument("--benchmark", default="niah-srt", choices=_ALLOWED_BENCHMARKS)
    p.add_argument("--policies", default="recency@50", help="e.g. 'recency@50,bm25@50'")
    p.add_argument("--budgets", default="16000", help="e.g. '1000,4000,16000'")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--log-dir", default="./logs")
    p.add_argument("--output", default=None)
    p.add_argument("--message-limit", type=int, default=None)

    p.add_argument("--niah-language", default="en")
    p.add_argument("--niah-min-context-words", type=int, default=16000)
    p.add_argument("--niah-max-context-words", type=int, default=16000)
    p.add_argument("--niah-n-contexts", type=int, default=1)
    p.add_argument("--niah-n-positions", type=int, default=3)
    p.add_argument("--niah-start-buffer-words", type=int, default=0)
    p.add_argument("--niah-end-buffer-words", type=int, default=0)
    p.add_argument("--niah-n-needles", type=int, default=10)
    p.add_argument("--niah-n-runs", type=int, default=1)
    p.add_argument("--niah-sample-method", default="sequential",
                   choices=["fixed", "sequential", "random"])
    p.add_argument("--niah-fixed-index", type=int, default=0)
    p.add_argument("--niah-chunk-words", type=int, default=200)
    p.add_argument("--niah-layout", default="query_first", choices=list(get_args(Layout)))
    p.add_argument("--niah-layouts", default=None,
                   help="Comma-separated layouts (overrides --niah-layout)")
    p.add_argument("--niah-preserve-system", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--niah-preserve-question", default=True, action=argparse.BooleanOptionalAction)

    p.add_argument("--resume", default=True, action=argparse.BooleanOptionalAction,
                   help="Skip cells that already have a success log in --log-dir.")
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        policies = parse_policies(args.policies)
        budgets = parse_budgets(args.budgets)
        layouts = parse_layouts(args.niah_layouts) if args.niah_layouts else [args.niah_layout]
    except ValueError as e:
        parser.error(str(e))

    print("Running evaluations:")
    print(f"  Model:     {args.model}")
    print(f"  Benchmark: {args.benchmark}")
    print(f"  Policies:  {[f'{p['name']}@{p['k']}' for p in policies]}")
    print(f"  Budgets:   {budgets}")
    print(f"  Layouts:   {layouts}")
    if args.limit:
        print(f"  Limit:     {args.limit} samples")

    logs = run_evaluations(
        policies=policies,
        budgets=budgets,
        model=args.model,
        limit=args.limit,
        log_dir=args.log_dir,
        benchmark=args.benchmark,
        layouts=layouts,
        niah_language=args.niah_language,
        niah_min_context_words=args.niah_min_context_words,
        niah_max_context_words=args.niah_max_context_words,
        niah_n_contexts=args.niah_n_contexts,
        niah_n_positions=args.niah_n_positions,
        niah_start_buffer_words=args.niah_start_buffer_words,
        niah_end_buffer_words=args.niah_end_buffer_words,
        niah_n_needles=args.niah_n_needles,
        niah_n_runs=args.niah_n_runs,
        niah_sample_method=args.niah_sample_method,
        niah_fixed_index=args.niah_fixed_index,
        niah_chunk_words=args.niah_chunk_words,
        niah_preserve_system=args.niah_preserve_system,
        niah_preserve_question=args.niah_preserve_question,
        message_limit=args.message_limit,
        resume=args.resume,
    )

    output_path = args.output
    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = f"{args.log_dir}/aggregated_{ts}.json"
    aggregate_results(logs, output_path)

    if not logs:
        print(f"\nNo eval logs were returned; see {args.log_dir!r}.", file=sys.stderr)
        sys.exit(1)

    all_ok = all(getattr(log, "status", None) == "success" for log in logs)
    print(f"\n  Eval log files: {len(logs)}")
    print(f"  Aggregated JSON: {output_path}")
    if not all_ok:
        print("One or more evaluations did not finish successfully. See messages above.",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
