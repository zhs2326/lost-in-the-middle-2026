#!/usr/bin/env python3
"""Plot the "lost in the middle" position curves from saved prediction files.

Scores the prediction JSONL(.gz) files written by `run_api_sweep.py` /
`get_*_responses_from_api.py` using the official metrics, then draws accuracy vs.
gold position for the multi-document QA and/or key-value retrieval tasks, with a
**bootstrap 95% confidence interval** error bar on every point (seeded, so the
figure is deterministic). At n=100 the bars are wide relative to the ~0.07-0.14
position spreads — which is the whole point: a curve whose error bars overlap
should *look* flat, not like a U.

Example:

    python ./scripts/plot_results.py --model claude-sonnet-4-6 \
        --output lost_in_the_middle_curve.png
"""
import argparse
import glob
import json
import logging
import os
import re
import statistics
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from xopen import xopen

from lost_in_the_middle.bootstrap import DEFAULT_B, DEFAULT_SEED
from lost_in_the_middle.metrics import best_subspan_em

matplotlib.use("Agg")
logger = logging.getLogger(__name__)


def safe_model_slug(model_name):
    return model_name.replace("/", "_").replace(":", "_")


def _gold_index_from_path(path):
    match = re.search(r"gold_at_(\d+)", path)
    return int(match.group(1)) if match else None


def _score_qa_file(path):
    """Paper-faithful metric: per-example best_subspan_em over the FIRST LINE."""
    scores = []
    with xopen(path) as fin:
        for line in fin:
            ex = json.loads(line)
            prediction = ex["model_answer"].split("\n")[0].strip()
            scores.append(best_subspan_em(prediction=prediction, ground_truths=ex["answers"]))
    return scores


def _score_qa_file_full(path):
    """Lenient metric: per-example best_subspan_em over the WHOLE answer."""
    scores = []
    with xopen(path) as fin:
        for line in fin:
            ex = json.loads(line)
            scores.append(best_subspan_em(prediction=ex["model_answer"], ground_truths=ex["answers"]))
    return scores


def _score_kv_file(path):
    scores = []
    with xopen(path) as fin:
        for line in fin:
            ex = json.loads(line)
            scores.append(1.0 if ex["value"].lower() in ex["model_answer"].lower() else 0.0)
    return scores


def _bootstrap_ci(scores, B=DEFAULT_B, seed=DEFAULT_SEED):
    """Percentile bootstrap 95% CI of the mean of 0/1 scores (seeded)."""
    arr = np.asarray(scores, dtype=float)
    n = len(arr)
    if n == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    reps = arr[idx].mean(axis=1)
    return float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))


def collect(pattern, score_fn):
    """Return ([gold_index...], [mean...], [ci_low...], [ci_high...], n_examples) sorted by gold index."""
    points = []
    n_examples = None
    for path in glob.glob(pattern):
        gold_index = _gold_index_from_path(path)
        if gold_index is None:
            continue
        scores = score_fn(path)
        n_examples = len(scores)
        mean = statistics.mean(scores) if scores else float("nan")
        lo, hi = _bootstrap_ci(scores)
        points.append((gold_index, mean, lo, hi))
    points.sort()
    if not points:
        return [], [], [], [], n_examples
    xs, ys, los, his = zip(*points)
    return list(xs), list(ys), list(los), list(his), n_examples


def _yerr(ys, los, his):
    """Asymmetric error-bar arrays (distance from mean down to ci_low / up to ci_high)."""
    lower = [y - lo for y, lo in zip(ys, los)]
    upper = [hi - y for y, hi in zip(ys, his)]
    return [lower, upper]


def main():
    logging.basicConfig(format="%(asctime)s - %(module)s - %(levelname)s - %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Model slug used when the predictions were written.")
    parser.add_argument("--output", default=None,
                        help="Output image path. Defaults to results/<slug>/lost_in_the_middle_curve_<slug>.png.")
    parser.add_argument("--title", default=None, help="Optional overall figure title.")
    parser.add_argument("--task", choices=["qa", "kv", "both"], default="both", help="Which task(s) to plot.")
    parser.add_argument("--num-documents", type=int, default=20, help="QA document-count setting to plot (10/20/30).")
    parser.add_argument("--num-keys", type=int, default=140, help="KV key-count setting to plot (75/140/300).")
    args = parser.parse_args()

    slug = safe_model_slug(args.model)
    output = args.output or f"results/{slug}/lost_in_the_middle_curve_{slug}.png"
    os.makedirs(os.path.dirname(output), exist_ok=True)
    qa_pattern = f"qa_predictions/{args.num_documents}_total_documents/*gold_at_*-{slug}-predictions.jsonl.gz"
    kv_pattern = f"kv_predictions/kv-retrieval-{args.num_keys}_keys_gold_at_*-{slug}-predictions.jsonl.gz"

    empty = ([], [], [], [], None)
    qa_x, qa_y, qa_lo, qa_hi, qa_n = collect(qa_pattern, _score_qa_file) if args.task in ("qa", "both") else empty
    qa_xf, qa_yf, qa_lof, qa_hif, _ = collect(qa_pattern, _score_qa_file_full) if args.task in ("qa", "both") else empty
    kv_x, kv_y, kv_lo, kv_hi, kv_n = collect(kv_pattern, _score_kv_file) if args.task in ("kv", "both") else empty

    panels = [p for p in [("qa", qa_x, qa_y, qa_lo, qa_hi, qa_n), ("kv", kv_x, kv_y, kv_lo, kv_hi, kv_n)] if p[1]]
    if not panels:
        logger.error("No prediction files found for model slug '%s'. Patterns:\n  %s\n  %s", slug, qa_pattern, kv_pattern)
        sys.exit(1)

    fig, axes = plt.subplots(1, len(panels), figsize=(6.4 * len(panels), 4.8), squeeze=False)
    for ax, (kind, xs, ys, los, his, n) in zip(axes[0], panels):
        if kind == "qa":
            # Paper-faithful (first-line) curve with bootstrap CI error bars.
            ax.errorbar(xs, ys, yerr=_yerr(ys, los, his), marker="o", linewidth=2, markersize=8,
                        color="#1f77b4", capsize=4, label="best_subspan_em (first line — paper metric)")
            # Lenient (full-answer) curve, if available.
            if qa_xf:
                ax.errorbar(qa_xf, qa_yf, yerr=_yerr(qa_yf, qa_lof, qa_hif), marker="s", linewidth=2, markersize=7,
                            color="#ff7f0e", linestyle="--", capsize=4,
                            label="best_subspan_em (full answer — lenient)")
            ax.set_title(f"Multi-document QA ({args.num_documents} docs, n={n}/pos)")
            ax.set_xlabel("Position of gold document in context")
            ax.set_ylabel("best_subspan_em")
            ax.legend(fontsize=8, loc="center right")
        else:
            ax.errorbar(xs, ys, yerr=_yerr(ys, los, his), marker="o", linewidth=2, markersize=8,
                        color="#1f77b4", capsize=4)
            ax.set_title(f"Key-value retrieval ({args.num_keys} keys, n={n}/pos)")
            ax.set_xlabel("Position of queried key in context")
            ax.set_ylabel("accuracy")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(xs)

    fig.suptitle(args.title or f"'Lost in the Middle' position curves (95% CI) — {args.model}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, dpi=150)
    logger.info("Wrote %s", output)
    # Also echo the underlying numbers with CIs.
    if qa_x:
        print("QA first-line:", {x: f"{y:.3f} [{lo:.3f},{hi:.3f}]" for x, y, lo, hi in zip(qa_x, qa_y, qa_lo, qa_hi)})
    if qa_xf:
        print("QA full:      ", {x: f"{y:.3f} [{lo:.3f},{hi:.3f}]" for x, y, lo, hi in zip(qa_xf, qa_yf, qa_lof, qa_hif)})
    if kv_x:
        print("KV accuracy:  ", {x: f"{y:.3f} [{lo:.3f},{hi:.3f}]" for x, y, lo, hi in zip(kv_x, kv_y, kv_lo, kv_hi)})


if __name__ == "__main__":
    main()
