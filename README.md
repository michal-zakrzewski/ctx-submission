# Layout Matters: Reproduction Code

Anonymous code release accompanying the NeurIPS 2026 Evaluations & Datasets
submission *"Layout Matters: How Prompt Structure Shapes Long-Context
Retrieval Across Model Families"*.

The repository contains the evidence-controlled layout protocol, the five
layout variants, the three context-selection policies, the NIAH-SRT and
HotpotQA task wrappers, and the aggregation pipeline used to produce the
tables and Figure 1 in the paper. No API keys, model weights, or cached
datasets are included — everything is fetched from public sources at run
time (HuggingFace for datasets, Ollama for local models).


## Repository layout

```
src/ctxlab/
  core/                 layout definitions + word-budget context manager
  policies/             recency, keyword, BM25 (message-granularity)
  inspect/
    tasks/niah_srt.py   single-needle retrieval (OpenCompass NeedleBench)
    tasks/hotpotqa.py   distractor HotpotQA
    solvers.py          two-stage evidence-selector + layout-solver chain
    scorers.py          F1, normalized EM, context-efficiency metric
  aggregate/            CLI that turns .eval logs into per-cell metric JSON
  eval/                 sweep CLI (policy x budget x layout, resumable)

scripts/
  export_rows.py        dump per-sample rows to JSONL
  plot_layout_gap.py    reproduce Figure 1 from the paper tables

tests/                  unit tests for layout and BM25
Makefile                one-line targets for each experiment
```


## Prerequisites

- Python 3.12+
- [`uv`](https://github.com/astral-sh/uv) for environment management (or any
  tool that understands `pyproject.toml`)
- [Ollama](https://ollama.com/) running locally, accessible at
  `http://localhost:11434` (the default). Any OpenAI-compatible endpoint
  will also work — pass the model id as `openai/<name>` to Inspect AI.
- Internet access the first time you run each task (HuggingFace downloads
  the NeedleBench haystack and HotpotQA distractor set).


## Install

```bash
uv sync
```

To install in a plain venv without `uv`:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```


## Downloading the models

All eight paper models are pulled through Ollama. Disk footprint is 4–6 GB
per model. Pulls can be resumed.

```bash
ollama pull deepseek-r1:1.5b
ollama pull llama3.2:3b
ollama pull ministral-3:3b             # needs recent Ollama (>=0.3.14)
ollama pull phi3.5:3.8b
ollama pull qwen2.5:3b
ollama pull gemma2:9b
ollama pull qwen2.5-coder:7b
ollama pull llama3.1:8b
```

Set `num_ctx` to 32768 on the Ollama side, or Inspect AI will cap context
at the model default (often 4k, which truncates the 16k-word haystack). The
simplest way is to create a variant tag once, e.g.

```bash
cat > Modelfile-qwen7b <<'EOF'
FROM qwen2.5-coder:7b
PARAMETER num_ctx 32768
EOF
ollama create qwen2.5-coder:7b-32k -f Modelfile-qwen7b
```

and then pass `--model ollama/qwen2.5-coder:7b-32k` to the eval CLI.


## Quick check

Runs one 400-word sample on the default model, to confirm that the Ollama
endpoint responds and the pipeline is wired correctly:

```bash
make smoke MODEL=ollama/qwen2.5-coder:7b
```

This writes logs under `logs/aggregated_<timestamp>.json` and should finish
in under a minute.


## Reproducing the main experiments

Each target writes `.eval` logs under `logs/<exp>/<model-tag>/` and is
**resumable** — re-running the same command skips cells that already have a
`status=success` log.

### RQ1: layout ablation (Table 1, Figure 1)

Five layouts x one policy (recency@50) x one budget (16k words),
`n=10` needles per cell. One model takes roughly 30–60 minutes on a single
consumer GPU (Qwen 7B on a 5090, layout `query_first` finishes in ~8
minutes; `baseline` and `isolated_sections` in ~20).

```bash
for m in \
    deepseek-r1:1.5b llama3.2:3b ministral-3:3b gemma2:9b \
    phi3.5:3.8b qwen2.5:3b qwen2.5-coder:7b llama3.1:8b; do
  make exp1 MODEL=ollama/$m
done
```

Aggregate each model's logs:

```bash
for m in deepseek-r1_1.5b llama3.2_3b ministral-3_3b gemma2_9b \
         phi3.5_3.8b qwen2.5_3b qwen2.5-coder_7b llama3.1_8b; do
  uv run python -m ctxlab.aggregate \
    --log-dir ./logs/exp1/$m --ci --group-by-depth \
    --output ./logs/exp1/$m/metrics.json
done
```

The `metrics.json` files contain per-cell F1, EM, normalised EM, selection
rate, conditional EM (EM given needle in selected evidence), and bootstrap
95% CIs. Table 1 in the paper reports the `depth_pct=100` rows.

### RQ2: policy x budget grid (Table 2)

Three policies (recency, keyword, BM25) x four budgets (1k, 4k, 8k, 16k) x
two layouts (baseline, query-sandwich). Runs on one model at a time:

```bash
make exp2 MODEL=ollama/qwen2.5-coder:7b
make aggregate EXP=exp2
```

### HotpotQA transfer (Table 3)

Three layouts x two policies x two budgets on the HotpotQA validation
split. The dataset is ~7,400 samples per cell; the recency@50 / budget 1k /
baseline cell takes about 2 hours on a 5090 with Llama 3.2 3B.

```bash
make exp-hotpotqa MODEL=ollama/llama3.2:3b
make aggregate EXP=hotpotqa
```


## Figure 1

`scripts/plot_layout_gap.py` reads the hardcoded numbers from Table 1 and
produces `fig_layout_gap.pdf`:

```bash
make figure
```

The numbers there are the same ones shipped in the paper; rerunning the
experiments above will give slightly different values (±0.03–0.05 F1 is
expected at `n=10` with default Ollama decoding).


## Determinism and variance

Inspect AI passes no explicit `temperature` to Ollama, which means each
model uses its own default (typically temperature 0.8, top-p 0.9). The
haystack itself — needle position, distractor sampling — is seeded from a
deterministic condition key, so context content reproduces exactly across
runs. Per-sample answers vary with decoding.

To make answers deterministic as well, pass `temperature=0` through
Inspect AI's generate config:

```bash
uv run python -m ctxlab.eval \
    --benchmark niah-srt --model ollama/llama3.2:3b \
    --policies recency@50 --budgets 16000 \
    --niah-layouts baseline,query_first,query_sandwich,q_repeat_end,isolated_sections \
    --niah-min-context-words 16000 --niah-max-context-words 16000 \
    --niah-n-contexts 1 --niah-n-positions 3 --niah-n-needles 10 \
    --niah-sample-method sequential --niah-chunk-words 200 \
    --model-config temperature=0 \
    --log-dir ./logs/exp1-temp0/llama3.2_3b
```

(We report stochastic-default numbers in the paper to match typical
deployment conditions; the Group A / B split holds under both settings.)


## Tests

A handful of unit tests cover layout message assembly and BM25 scoring:

```bash
make test
```


## Licence

The paper and this code release are submitted under the NeurIPS 2026
Evaluations & Datasets track. A permissive licence (MIT / Apache-2.0) will
accompany the de-anonymised release after review.
