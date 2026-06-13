#!/usr/bin/env python3
"""Multi-document QA position curves for one model, across context lengths.

Recreates the original *Lost in the Middle* figure (Liu et al., arXiv:2307.03172)
for a single model: accuracy (`best_subspan_em`) vs. the relative position of the
gold document, with one line per context-length setting (10/20/30/... documents).
Two panels show the paper-faithful **first-line** metric and the lenient
**full-answer** metric; oracle (upper bound) and closed-book (parametric-memory
floor) are drawn as horizontal reference lines.

Reads the aggregate summary CSVs written by `summarize_qa_results.py`
(`results/<slug>/qa_<N>docs_<slug>_summary.csv`). If `--num-documents` is omitted,
every context length found on disk for the model is plotted — so the figure always
shows the full scaling story and never a redundant subset.

Example:

    python ./scripts/plot_qa_comparison.py --model deepseek:deepseek-chat
"""
import argparse
import csv
import glob
import logging
import os
import re
import sys

import matplotlib
import matplotlib.pyplot as plt

from lost_in_the_middle.plotting import (
    CLOSEDBOOK_STYLE,
    DOC_COUNT_COLORS,
    ORACLE_STYLE,
    apply_paper_style,
    fallback_color_iter,
)

matplotlib.use("Agg")
logger = logging.getLogger(__name__)


def safe_model_slug(model_name):
    return model_name.replace("/", "_").replace(":", "_")


def discover_document_counts(slug):
    """Return every QA context-length setting that has a summary CSV on disk."""
    counts = []
    for path in glob.glob(f"results/{slug}/qa_*docs_{slug}_summary.csv"):
        match = re.search(r"qa_(\d+)docs_", os.path.basename(path))
        if match:
            counts.append(int(match.group(1)))
    return sorted(set(counts))


def load_baselines(slug):
    """Return {'oracle': {...}, 'closedbook': {...}} or {} if the file is absent."""
    path = f"results/{slug}/qa_baselines_{slug}_summary.csv"
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["condition"]] = {
                "first": float(row["best_subspan_em_first_line"]),
                "full": float(row["best_subspan_em_full"]),
            }
    return out


def load_summary(slug, num_documents):
    path = f"results/{slug}/qa_{num_documents}docs_{slug}_summary.csv"
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    rows.sort(key=lambda r: int(r["gold_index"]))
    gold = [int(r["gold_index"]) for r in rows]
    rel = [g / (num_documents - 1) for g in gold]  # 0..1
    first = [float(r["best_subspan_em_first_line"]) for r in rows]
    full = [float(r["best_subspan_em_full"]) for r in rows]
    n = rows[0]["n_examples"] if rows else "?"
    # Bootstrap 95% CI bounds, when the summary carries them (older summaries may not).
    def _band(lo_col, hi_col):
        if rows and lo_col in rows[0] and hi_col in rows[0]:
            return [float(r[lo_col]) for r in rows], [float(r[hi_col]) for r in rows]
        return None
    first_ci = _band("em_first_line_ci_low", "em_first_line_ci_high")
    full_ci = _band("em_full_ci_low", "em_full_ci_high")
    return rel, first, full, n, first_ci, full_ci


def main():
    logging.basicConfig(format="%(asctime)s - %(module)s - %(levelname)s - %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-documents", type=int, nargs="+", default=None,
                        help="Context lengths to plot. Default: every setting found on disk for the model.")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    slug = safe_model_slug(args.model)
    document_counts = args.num_documents or discover_document_counts(slug)
    if not document_counts:
        logger.error("No QA summary CSVs found for '%s' under results/%s/.", args.model, slug)
        sys.exit(1)
    output = args.output or f"results/{slug}/qa_comparison_{slug}.png"
    os.makedirs(os.path.dirname(output), exist_ok=True)

    apply_paper_style()
    fig, (ax_first, ax_full) = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
    fallback = fallback_color_iter()

    for num_documents in document_counts:
        rel, first, full, n, first_ci, full_ci = load_summary(slug, num_documents)
        color = DOC_COUNT_COLORS.get(num_documents) or next(fallback, "#333333")
        label = f"{num_documents} documents"
        ax_first.plot(rel, first, marker="o", color=color, label=label)
        ax_full.plot(rel, full, marker="o", color=color, label=label)
        # Shade the bootstrap 95% CI so overlapping bands visibly read as "flat / noisy".
        if first_ci is not None:
            ax_first.fill_between(rel, first_ci[0], first_ci[1], color=color, alpha=0.15, linewidth=0)
        if full_ci is not None:
            ax_full.fill_between(rel, full_ci[0], full_ci[1], color=color, alpha=0.15, linewidth=0)

    # Oracle (upper bound) and closed-book (parametric-memory floor) reference lines.
    baselines = load_baselines(slug)
    ref = [("oracle", "oracle (gold doc only)", ORACLE_STYLE),
           ("closedbook", "closed-book (no docs)", CLOSEDBOOK_STYLE)]
    for ax, key in [(ax_first, "first"), (ax_full, "full")]:
        for cond, ref_label, style in ref:
            if cond in baselines:
                ax.axhline(baselines[cond][key], label=ref_label, **style)

    for ax, title in [
        (ax_first, "First-line answer (paper-faithful)"),
        (ax_full, "Full answer (lenient)"),
    ]:
        ax.set_title(title)
        ax.set_xlabel("Position of gold document\n(0 = first  →  1 = last)")
        ax.set_ylim(0, 1.0)
        ax.set_xlim(-0.03, 1.03)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_first.set_ylabel("Accuracy (best_subspan_em)")

    # Single shared legend below the panels keeps both plots uncluttered.
    handles, labels = ax_first.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5),
               bbox_to_anchor=(0.5, -0.02), frameon=True)

    fig.suptitle(f"Lost in the Middle — multi-document QA across context lengths\n{args.model}",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(output)
    logger.info("Wrote %s  (context lengths: %s)", output, ", ".join(map(str, document_counts)))


if __name__ == "__main__":
    main()
