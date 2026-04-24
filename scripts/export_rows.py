"""Dump per-sample rows from Inspect eval logs to JSONL.

Produces the minimum schema used for downstream analysis
(depth filtering, paired bootstrap, per-layout comparisons).

    python scripts/export_rows.py --log-dir ./logs/exp1 --output data/rows.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ctxlab.aggregate.__main__ import _extract_sample_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", default="./logs")
    parser.add_argument("--output", default="rows.jsonl")
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

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows = n_ok = n_skip = 0
    with out_path.open("w") as f:
        for info in infos:
            log = read_eval_log(info)
            if log.status != "success":
                n_skip += 1
                continue
            n_ok += 1
            for row in _extract_sample_rows(log):
                f.write(json.dumps(row) + "\n")
                n_rows += 1

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {out_path}: {n_rows} rows from {n_ok} logs ({n_skip} skipped), {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
