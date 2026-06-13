#!/usr/bin/env python3
"""Summarize multi-document QA prediction files into tidy CSV/JSON for graphing.

Reads the prediction JSONL(.gz) files written by the API scripts for one model and
one document-count setting, and writes:

- a **per-example** CSV (one row per question/position) with both metrics, and
- an **aggregate** CSV + JSON (one row per gold position) with **bootstrap 95%
  confidence intervals**.

Two metrics are recorded:
- `em_first_line` — the PAPER-FAITHFUL metric: `best_subspan_em` on the first line
  of the answer (matches `evaluate_qa_responses.py`).
- `em_full`       — a LENIENT variant: `best_subspan_em` over the whole answer.
  Useful because modern chat models often add a Markdown heading / preamble on the
  first line, which can deflate the first-line metric even when the answer is right.

**Confidence intervals.** Each position's score is the mean of n binary outcomes —
a noisy proportion at n=100 — while the project's headline shapes rest on
between-position spreads of only ~0.07-0.14. So a bare mean cannot be distinguished
from sampling noise. This script adds, per position, a seeded percentile bootstrap
95% CI for both metrics, and a **paired** bootstrap of the difference vs a reference
position (default: the first/start gold index) — the actual test of whether a
position effect is real (CI excludes 0) or a non-result (CI straddles 0). A spread
CI (best minus worst position) is written to the JSON as a one-number summary. See
`CI_IMPLEMENTATION_HANDOFF.md` and `src/lost_in_the_middle/bootstrap.py`.

Example:

    python ./scripts/summarize_qa_results.py --model claude-sonnet-4-6 \
        --num-documents 10 --gold-indices 0 4 9 --outdir results
"""
import argparse
import csv
import glob
import json
import logging
import os

import numpy as np
from xopen import xopen

from lost_in_the_middle.bootstrap import DEFAULT_B, DEFAULT_SEED, position_bootstrap
from lost_in_the_middle.metrics import best_subspan_em

logger = logging.getLogger(__name__)


def safe_model_slug(model_name):
    return model_name.replace("/", "_").replace(":", "_")


def find_prediction_file(num_documents, gold_index, slug):
    pattern = f"qa_predictions/{num_documents}_total_documents/*gold_at_{gold_index}-{slug}-predictions.jsonl.gz"
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No prediction file matched: {pattern}")
    return sorted(matches)[0]


def read_position(path):
    """Score one prediction file, keeping per-example detail aligned in file order.

    Returns ``(questions, first_scores, full_scores, n_failed, examples)`` where
    ``examples`` carries the extra per-row fields the per-example CSV records.
    """
    questions, first_scores, full_scores, examples = [], [], [], []
    n_failed = 0
    with xopen(path) as fin:
        for line in fin:
            ex = json.loads(line)
            answer = ex["model_answer"]
            failed = ex.get("model_answer_failed", False)
            n_failed += 1 if failed else 0
            first_line = answer.split("\n")[0].strip()
            em_first = best_subspan_em(prediction=first_line, ground_truths=ex["answers"])
            em_full = best_subspan_em(prediction=answer, ground_truths=ex["answers"])
            questions.append(ex.get("question", ""))
            first_scores.append(em_first)
            full_scores.append(em_full)
            examples.append({
                "gold_answers": "|".join(ex.get("answers", [])),
                "model_answer_failed": failed,
                "model_answer": answer.replace("\n", " ").strip(),
            })
    return questions, first_scores, full_scores, n_failed, examples


def main():
    logging.basicConfig(format="%(asctime)s - %(module)s - %(levelname)s - %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-documents", type=int, required=True)
    parser.add_argument("--gold-indices", type=int, nargs="+", required=True)
    parser.add_argument("--outdir", default="results")
    parser.add_argument("--ref-gold-index", type=int, default=None,
                        help="Reference position for paired-difference CIs. Default: the first (smallest) gold index.")
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_B)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    slug = safe_model_slug(args.model)
    outdir = os.path.join(args.outdir, slug)
    os.makedirs(outdir, exist_ok=True)
    base = f"{outdir}/qa_{args.num_documents}docs_{slug}"

    # --- Read every position, keeping per-example scores aligned for the paired bootstrap. ---
    per_position = {}
    per_example_rows = []
    for gold_index in args.gold_indices:
        path = find_prediction_file(args.num_documents, gold_index, slug)
        questions, first_scores, full_scores, n_failed, examples = read_position(path)
        per_position[gold_index] = {
            "questions": questions, "first": first_scores, "full": full_scores, "n_failed": n_failed
        }
        for q, em_first, em_full, extra in zip(questions, first_scores, full_scores, examples):
            per_example_rows.append({
                "model": args.model,
                "num_documents": args.num_documents,
                "gold_index": gold_index,
                "question": q,
                "gold_answers": extra["gold_answers"],
                "em_first_line": em_first,
                "em_full": em_full,
                "model_answer_failed": extra["model_answer_failed"],
                "model_answer": extra["model_answer"],
            })

    gold_order = sorted(per_position)
    ref = args.ref_gold_index if args.ref_gold_index is not None else gold_order[0]
    if ref not in per_position:
        parser.error(f"--ref-gold-index {ref} is not among the scored positions {gold_order}.")

    # The same questions appear at every gold position (only the gold doc index moves),
    # so the i-th example is the same question everywhere -> align by index. Truncate to
    # the shortest position and warn if the question sequences disagree (they shouldn't).
    n = min(len(per_position[g]["questions"]) for g in gold_order)
    ref_questions = per_position[gold_order[0]]["questions"][:n]
    for g in gold_order:
        if per_position[g]["questions"][:n] != ref_questions:
            logger.warning("Question order at gold=%d differs from the reference order; "
                           "paired CIs assume index alignment — verify the input files.", g)
    scores_first = {g: np.asarray(per_position[g]["first"][:n], dtype=float) for g in gold_order}
    scores_full = {g: np.asarray(per_position[g]["full"][:n], dtype=float) for g in gold_order}

    ci_first, spread_first = position_bootstrap(scores_first, gold_order, ref,
                                                B=args.bootstrap_samples, seed=args.seed)
    ci_full, spread_full = position_bootstrap(scores_full, gold_order, ref,
                                              B=args.bootstrap_samples, seed=args.seed)

    aggregate = []
    for g in gold_order:
        row = {
            "model": args.model,
            "num_documents": args.num_documents,
            "gold_index": g,
            "n_examples": n,
            "n_failed": per_position[g]["n_failed"],
            "best_subspan_em_first_line": ci_first[g]["mean"],
            "best_subspan_em_full": ci_full[g]["mean"],
            "em_first_line_ci_low": ci_first[g]["ci_low"],
            "em_first_line_ci_high": ci_first[g]["ci_high"],
            "em_full_ci_low": ci_full[g]["ci_low"],
            "em_full_ci_high": ci_full[g]["ci_high"],
            "ref_gold_index": ref,
            "em_first_line_diff_vs_ref": ci_first[g].get("diff_vs_ref", 0.0),
            "em_first_line_diff_ci_low": ci_first[g].get("diff_ci_low", 0.0),
            "em_first_line_diff_ci_high": ci_first[g].get("diff_ci_high", 0.0),
            "em_first_line_diff_excludes_zero": ci_first[g].get("diff_excludes_zero", False),
            "em_full_diff_vs_ref": ci_full[g].get("diff_vs_ref", 0.0),
            "em_full_diff_ci_low": ci_full[g].get("diff_ci_low", 0.0),
            "em_full_diff_ci_high": ci_full[g].get("diff_ci_high", 0.0),
            "em_full_diff_excludes_zero": ci_full[g].get("diff_excludes_zero", False),
        }
        aggregate.append(row)
        # ASCII-only log line (Windows console is GBK; non-ASCII raises).
        diff_note = ""
        if g != ref:
            diff_note = "  d_full=%+.3f [%+.3f, %+.3f]%s" % (
                ci_full[g]["diff_vs_ref"], ci_full[g]["diff_ci_low"], ci_full[g]["diff_ci_high"],
                " SIG" if ci_full[g]["diff_excludes_zero"] else "")
        logger.info("gold=%d n=%d  full=%.3f [%.3f, %.3f]  first=%.3f [%.3f, %.3f]%s",
                    g, n, ci_full[g]["mean"], ci_full[g]["ci_low"], ci_full[g]["ci_high"],
                    ci_first[g]["mean"], ci_first[g]["ci_low"], ci_first[g]["ci_high"], diff_note)
    logger.info("spread(full) best@%d - worst@%d = %.3f [%.3f, %.3f] %s",
                spread_full["best_position"], spread_full["worst_position"], spread_full["spread"],
                spread_full["ci_low"], spread_full["ci_high"],
                "SIGNIFICANT" if spread_full["excludes_zero"] else "n.s. (overlaps 0)")

    # Write per-example CSV.
    per_example_path = f"{base}_per_example.csv"
    with open(per_example_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_example_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_example_rows)

    # Write aggregate CSV.
    summary_csv_path = f"{base}_summary.csv"
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregate[0].keys()))
        writer.writeheader()
        writer.writerows(aggregate)

    # Write aggregate JSON (adds bootstrap metadata + spread CIs alongside per-position rows).
    summary_json_path = f"{base}_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model,
            "task": "multi_document_qa",
            "num_documents": args.num_documents,
            "bootstrap": {"B": args.bootstrap_samples, "seed": args.seed,
                          "ref_gold_index": ref, "n_per_position": n},
            "spread_full": spread_full,
            "spread_first_line": spread_first,
            "results": aggregate,
        }, f, indent=2)

    logger.info("Wrote:\n  %s\n  %s\n  %s", per_example_path, summary_csv_path, summary_json_path)


if __name__ == "__main__":
    main()
