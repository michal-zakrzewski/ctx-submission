"""Aggregate Inspect AI eval logs into per-cell metric rows.

    python -m ctxlab.aggregate --log-dir ./logs --ci --group-by-depth
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


_LETTER_MAP = {"C": 1.0, "I": 0.0}


def _extract_sample_rows(log) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not log.samples:
        return rows

    task_args = dict(getattr(log.eval, "task_args", {}) or {})
    common = {
        "model": str(log.eval.model) if log.eval.model else "unknown",
        "task": str(log.eval.task) if log.eval.task else "unknown",
        "policy": task_args.get("policy_name", "unknown"),
        "policy_k": task_args.get("policy_k", 0),
        "token_budget": task_args.get("token_budget", 0),
        "layout": task_args.get("layout", "unknown"),
    }

    for sample in log.samples:
        meta = dict(sample.metadata) if sample.metadata else {}
        row: dict[str, Any] = {
            "sample_id": str(sample.id) if sample.id else None,
            **common,
            "depth_pct": meta.get("depth_pct"),
            "context_words": meta.get("context_words"),
            "needle_in_evidence": meta.get("needle_in_evidence"),
        }

        if sample.scores:
            for scorer_name, score_obj in sample.scores.items():
                v = score_obj.value
                if isinstance(v, (int, float)):
                    row[scorer_name] = v
                elif isinstance(v, str):
                    if v in _LETTER_MAP:
                        row[scorer_name] = _LETTER_MAP[v]
                    else:
                        try:
                            row[scorer_name] = float(v)
                        except ValueError:
                            pass

        rows.append(row)

    return rows


def _group_key(row: dict[str, Any], *, group_by_depth: bool) -> tuple:
    key = (
        row.get("model", ""),
        row.get("policy", ""),
        row.get("policy_k", 0),
        row.get("token_budget", 0),
        row.get("layout", ""),
    )
    if group_by_depth:
        key = (*key, row.get("depth_pct"))
    return key


def _bootstrap_ci(
    values: list[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    if len(values) < 2:
        m = values[0] if values else 0.0
        return (m, m)
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        sample = [rng.choice(values) for _ in range(len(values))]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo = means[int(n_boot * alpha / 2)]
    hi = means[int(n_boot * (1 - alpha / 2))]
    return (lo, hi)


def aggregate_rows(
    rows: list[dict[str, Any]],
    ci: bool = False,
    group_by_depth: bool = False,
) -> list[dict[str, Any]]:
    """Group rows by (model, policy, k, budget, layout[, depth_pct]) and compute metrics.

    For each cell we report: EM, F1, normalised EM, selection rate
    (fraction of samples where the needle was in the selected evidence),
    and conditional EM / conditional normalised EM (EM on samples where
    the needle was selected).
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[_group_key(row, group_by_depth=group_by_depth)].append(row)

    results = []
    for key, group in sorted(groups.items()):
        if group_by_depth:
            model, policy, policy_k, budget, layout, depth_pct = key
        else:
            model, policy, policy_k, budget, layout = key
            depth_pct = None

        em_vals = [r["exact"] for r in group if "exact" in r]
        f1_vals = [r["f1"] for r in group if "f1" in r]
        norm_em_vals = [r["normalized_exact"] for r in group if "normalized_exact" in r]

        needle_flags = [r["needle_in_evidence"] for r in group if r.get("needle_in_evidence") is not None]
        selected = sum(1 for f in needle_flags if f is True)
        total_flags = len(needle_flags)

        cond_em = [r["exact"] for r in group if r.get("needle_in_evidence") is True and "exact" in r]
        cond_norm_em = [
            r["normalized_exact"]
            for r in group
            if r.get("needle_in_evidence") is True and "normalized_exact" in r
        ]

        entry: dict[str, Any] = {
            "model": model,
            "policy": f"{policy}@{policy_k}",
            "token_budget": budget,
            "layout": layout,
        }
        if group_by_depth:
            entry["depth_pct"] = depth_pct
        entry.update({
            "n": len(group),
            "em_mean": sum(em_vals) / len(em_vals) if em_vals else None,
            "f1_mean": sum(f1_vals) / len(f1_vals) if f1_vals else None,
            "norm_em_mean": sum(norm_em_vals) / len(norm_em_vals) if norm_em_vals else None,
            "selection_rate": selected / total_flags if total_flags > 0 else None,
            "conditional_em": sum(cond_em) / len(cond_em) if cond_em else None,
            "conditional_norm_em": sum(cond_norm_em) / len(cond_norm_em) if cond_norm_em else None,
        })

        if ci and em_vals:
            entry["em_ci_95"] = _bootstrap_ci(em_vals)
        if ci and f1_vals:
            entry["f1_ci_95"] = _bootstrap_ci(f1_vals)
        if ci and norm_em_vals:
            entry["norm_em_ci_95"] = _bootstrap_ci(norm_em_vals)

        results.append(entry)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Inspect eval logs into metrics JSON.")
    parser.add_argument("--log-dir", default="./logs")
    parser.add_argument("--output", default=None, help="Output JSON path (default: stdout)")
    parser.add_argument("--ci", action="store_true", help="Include bootstrap 95%% CIs")
    parser.add_argument("--group-by-depth", action="store_true", help="Add depth_pct to the group key")
    args = parser.parse_args()

    from inspect_ai.log import list_eval_logs, read_eval_log

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"Log directory not found: {log_dir}", file=sys.stderr)
        sys.exit(1)

    infos = list_eval_logs(str(log_dir))
    if not infos:
        print(f"No eval logs found in {log_dir}", file=sys.stderr)
        sys.exit(1)

    rows: list[dict[str, Any]] = []
    for info in infos:
        log = read_eval_log(info)
        if log.status != "success":
            continue
        rows.extend(_extract_sample_rows(log))

    if not rows:
        print("No sample data found in successful logs.", file=sys.stderr)
        sys.exit(1)

    results = aggregate_rows(rows, ci=args.ci, group_by_depth=args.group_by_depth)
    output = {
        "total_samples": len(rows),
        "total_groups": len(results),
        "groups": results,
    }
    text = json.dumps(output, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text)
        print(f"Wrote {args.output} ({len(results)} groups, {len(rows)} samples)")
    else:
        print(text)


if __name__ == "__main__":
    main()
