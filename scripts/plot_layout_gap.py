"""Reproduce Figure 1 of the paper.

Per-model bar chart of depth=100%-conditioned token F1 across the five layouts,
with Group A (layout-sensitive) / Group B (layout-robust) split colour-coded.

Numbers are taken directly from the aggregated metrics used for Table 1. Run
    python scripts/plot_layout_gap.py
to regenerate the figure as ``fig_layout_gap.pdf`` in the current directory.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

LAYOUTS = ["baseline", "query_first", "query_sandwich", "q_repeat_end", "isolated_sections"]
LAYOUT_LABELS = ["Baseline", "Query-first", "Query-sandwich", "Q-repeat-end", "Iso-sections"]

# (is_collapse_group, F1 at depth=100% conditioned on needle in E' for each layout)
# Canonical sources: SLMs from rows.jsonl filtered to /exp1-rev/;
# Phi/Qwen3B/Gemma from logs/exp1/<model>/aggregated.json;
# Qwen2.5-Coder 7B and Llama 3.1 8B from rows_7b_*.jsonl.
DATA = {
    "DeepSeek-R1\n1.5B":      (True,  [0.445, 0.033, 0.321, 0.376, 0.496]),
    "Llama 3.2\n3B":          (True,  [0.693, 0.031, 0.797, 0.640, 0.887]),
    "Ministral 3\n3B":        (True,  [0.832, 0.029, 0.690, 0.504, 0.898]),
    "Gemma2\n9B":             (True,  [0.746, 0.027, 0.885, 0.723, 0.763]),
    "Phi-3.5-mini\n3.8B":     (False, [0.635, 0.815, 0.832, 0.580, 0.790]),
    "Qwen2.5\n3B":            (False, [1.000, 0.936, 1.000, 1.000, 1.000]),
    "Qwen2.5-Coder\n7B":      (False, [0.937, 0.974, 0.999, 0.995, 0.972]),
    "Llama 3.1\n8B":          (False, [0.732, 0.981, 0.962, 0.643, 0.651]),
}

COLOR_COLLAPSE = "#d62728"
COLOR_NONCOLLAPSE = "#1f77b4"
LAYOUT_COLORS = ["#555555", "#e07b39", "#2ca02c", "#9467bd", "#17becf"]


def main(out_path: str = "fig_layout_gap.pdf") -> None:
    models = list(DATA.keys())
    n_layouts = len(LAYOUTS)

    fig, ax = plt.subplots(figsize=(10, 4.2))

    bar_width = 0.14
    group_gap = 0.25
    group_width = n_layouts * bar_width

    x_centers = []
    for i, model in enumerate(models):
        group_start = i * (group_width + group_gap)
        _, vals = DATA[model]
        for j, color in enumerate(LAYOUT_COLORS):
            x = group_start + j * bar_width
            ax.bar(x, vals[j], width=bar_width * 0.92, color=color, alpha=0.85,
                   edgecolor="white", linewidth=0.4)
        x_centers.append(group_start + group_width / 2 - bar_width / 2)

    ax.set_xticks(x_centers)
    ax.set_xticklabels(models, fontsize=8.5)
    for tick, model in zip(ax.get_xticklabels(), models):
        tick.set_color(COLOR_COLLAPSE if DATA[model][0] else COLOR_NONCOLLAPSE)

    ax.set_ylabel("Token F1 (depth 100%, conditioned)", fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_xlim(-0.2, x_centers[-1] + group_width)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Collapse / non-collapse divider + header labels
    div_x = (x_centers[3] + x_centers[4]) / 2
    ax.axvline(div_x, color="gray", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.text(div_x - 0.55, 1.04, "Collapse group (query-first F1<0.05)",
            ha="center", va="bottom", fontsize=8, color=COLOR_COLLAPSE)
    ax.text(div_x + 0.72, 1.04, "Non-collapse group",
            ha="center", va="bottom", fontsize=8, color=COLOR_NONCOLLAPSE)

    layout_patches = [
        mpatches.Patch(color=c, label=lbl, alpha=0.85)
        for c, lbl in zip(LAYOUT_COLORS, LAYOUT_LABELS)
    ]
    ax.legend(handles=layout_patches, fontsize=7.5, ncol=5,
              loc="lower center", bbox_to_anchor=(0.5, -0.22),
              frameon=True, framealpha=0.9, edgecolor="lightgray")

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
