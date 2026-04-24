.PHONY: smoke exp1 exp2 exp-hotpotqa aggregate rows figure test lint format

# Model served over Ollama. Override at the CLI:
#   make exp1 MODEL=ollama/llama3.2:3b
MODEL ?= ollama/qwen2.5-coder:7b
MODEL_TAG = $(subst /,_,$(subst :,_,$(patsubst ollama/%,%,$(MODEL))))

# Shared NIAH-SRT defaults (paper Section 4.2):
#   16k-word context, 200-word chunks, 3 depths (0/50/100%), 10 needles per cell.
PAPER_NIAH_ARGS = \
	--niah-min-context-words 16000 \
	--niah-max-context-words 16000 \
	--niah-n-contexts 1 \
	--niah-n-positions 3 \
	--niah-n-needles 10 \
	--niah-sample-method sequential \
	--niah-chunk-words 200

# Quick pipeline check: one 400-word sample with recency@50.
smoke:
	uv run python -m ctxlab.eval \
		--benchmark niah-srt --model $(MODEL) \
		--policies recency@50 --budgets 4000 --limit 1 \
		--niah-min-context-words 400 --niah-max-context-words 400 \
		--niah-n-contexts 1 --niah-n-positions 1 --niah-chunk-words 100

# RQ1: layout ablation. Fix recency@50 + 16k budget, sweep all five layouts.
#   make exp1 MODEL=ollama/llama3.2:3b
exp1:
	uv run python -m ctxlab.eval \
		--benchmark niah-srt --model $(MODEL) \
		--policies recency@50 --budgets 16000 \
		--niah-layouts baseline,query_first,query_sandwich,q_repeat_end,isolated_sections \
		$(PAPER_NIAH_ARGS) \
		--log-dir ./logs/exp1/$(MODEL_TAG)

# RQ2: policy x budget grid on two representative layouts.
exp2:
	uv run python -m ctxlab.eval \
		--benchmark niah-srt --model $(MODEL) \
		--policies recency@50,keyword@50,bm25@50 \
		--budgets 1000,4000,8000,16000 \
		--niah-layouts baseline,query_sandwich \
		$(PAPER_NIAH_ARGS) \
		--log-dir ./logs/exp2/$(MODEL_TAG)

# HotpotQA (distractor) transfer.
exp-hotpotqa:
	uv run python -m ctxlab.eval \
		--benchmark hotpotqa --model $(MODEL) \
		--policies recency@50,bm25@50 \
		--budgets 1000,4000 \
		--niah-layouts baseline,query_sandwich,isolated_sections \
		--log-dir ./logs/hotpotqa/$(MODEL_TAG)

# Aggregate one experiment directory into a single metrics JSON (with bootstrap CIs
# and per-depth breakdowns).
#   make aggregate EXP=exp1
EXP ?= exp1
aggregate:
	uv run python -m ctxlab.aggregate \
		--log-dir ./logs/$(EXP) --ci --group-by-depth \
		--output ./logs/$(EXP)/metrics.json

# Dump per-sample rows for downstream analysis.
rows:
	uv run python scripts/export_rows.py \
		--log-dir ./logs/$(EXP) --output ./data/rows_$(EXP).jsonl

# Reproduce Figure 1.
figure:
	uv run python scripts/plot_layout_gap.py

test:
	uv run pytest -q

lint:
	uv run ruff check .

format:
	uv run ruff format .
