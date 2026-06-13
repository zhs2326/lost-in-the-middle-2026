#!/usr/bin/env python3
"""Overlay the QA position curve of *several models* for one document-count setting.

Where `plot_qa_comparison.py` overlays document-counts for a single model, this
overlays **models** at a fixed document-count — the cross-model "lost in the
middle" view. Reads each model's aggregate summary written by
`summarize_qa_results.py` (`results/<slug>/qa_<N>docs_<slug>_summary.csv`) and, if
present, its closed-book floor from `qa_baselines_<slug>_summary.csv`.

The x-axis is the gold document's *relative* position (0 = start, 1 = end). The
full-answer metric is plotted by default (modern chat models' Markdown preambles
deflate the first-line metric unevenly across models, so the lenient metric is the
fairer cross-model comparison); pass `--metric first_line` for the paper metric.

Example:

    python ./scripts/plot_model_comparison.py \
        --models gpt-4.1 deepseek:deepseek-chat gemini-2.5-flash claude-sonnet-4-6 \
        --num-documents 30 --output results/_cross_model/qa_crossmodel_30docs.png
"""
import argparse
import csv
import logging
import os

import matplotlib
import matplotlib.pyplot as plt

from lost_in_the_middle.plotting import MODEL_COLORS, apply_paper_style, fallback_color_iter

matplotlib.use("Agg")
logger = logging.getLogger(__name__)


def safe_model_slug(model_name):
    return model_name.replace("/", "_").replace(":", "_")


def load_summary(slug, num_documents, metric_col, ci_low_col, ci_high_col):
    path = f"results/{slug}/qa_{num_documents}docs_{slug}_summary.csv"
    if not os.path.exists(path):
        logger.warning("Missing summary for %s (%s) — skipping.", slug, path)
        return None
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: int(r["gold_index"]))
    rel = [int(r["gold_index"]) / (num_documents - 1) for r in rows]
    y = [float(r[metric_col]) for r in rows]
    n = rows[0]["n_examples"] if rows else "?"
    # Bootstrap 95% CI band, when present (older summaries may predate it).
    ci = None
    if rows and ci_low_col in rows[0] and ci_high_col in rows[0]:
        ci = ([float(r[ci_low_col]) for r in rows], [float(r[ci_high_col]) for r in rows])
    return rel, y, n, ci


def load_closedbook(slug, metric_key):
    path = f"results/{slug}/qa_baselines_{slug}_summary.csv"
    if not os.path.exists(path):
        return None
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["condition"] == "closedbook":
                return float(row[f"best_subspan_em_{metric_key}"])
    return None


def main():
    logging.basicConfig(format="%(asctime)s - %(module)s - %(levelname)s - %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", required=True, help="Model names (provider-prefixed as usual).")
    parser.add_argument("--num-documents", type=int, required=True)
    parser.add_argument("--metric", choices=["full", "first_line"], default="full")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    metric_col = "best_subspan_em_full" if args.metric == "full" else "best_subspan_em_first_line"
    metric_key = "full" if args.metric == "full" else "first_line"
    ci_low_col, ci_high_col = f"em_{metric_key}_ci_low", f"em_{metric_key}_ci_high"
    output = args.output or f"results/_cross_model/qa_crossmodel_{args.num_documents}docs.png"
    os.makedirs(os.path.dirname(output), exist_ok=True)

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    fallback = fallback_color_iter()
    for model in args.models:
        slug = safe_model_slug(model)
        color = MODEL_COLORS.get(slug) or next(fallback, "#333333")
        loaded = load_summary(slug, args.num_documents, metric_col, ci_low_col, ci_high_col)
        if loaded is None:
            continue
        rel, y, n, ci = loaded
        ax.plot(rel, y, marker="o", color=color, label=f"{model} (n={n})")
        # Shaded bootstrap 95% CI band — overlapping bands across models signal that
        # the "model-specific signatures" are within noise at this n.
        if ci is not None:
            ax.fill_between(rel, ci[0], ci[1], color=color, alpha=0.13, linewidth=0)
        floor = load_closedbook(slug, metric_key)
        if floor is not None:
            ax.axhline(floor, color=color, linewidth=1.2, linestyle=":", alpha=0.6)

    ax.set_title(
        f"Lost in the Middle across models — {args.num_documents}-document QA",
    )
    ax.set_xlabel("Position of gold document  (0 = first  →  1 = last)")
    ax.set_ylabel(f"Accuracy (best_subspan_em, {args.metric.replace('_', ' ')})")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(-0.03, 1.03)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(loc="lower left", title="Model (dotted = its closed-book floor)", title_fontsize=10)

    fig.tight_layout()
    fig.savefig(output)
    logger.info("Wrote %s", output)


if __name__ == "__main__":
    main()
