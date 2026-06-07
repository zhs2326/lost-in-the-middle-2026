#!/usr/bin/env python3
"""Summarize multi-document QA prediction files into tidy CSV/JSON for graphing.

Reads the prediction JSONL(.gz) files written by the API scripts for one model and
one document-count setting, and writes:

- a **per-example** CSV (one row per question/position) with both metrics, and
- an **aggregate** CSV + JSON (one row per gold position).

Two metrics are recorded:
- `em_first_line` — the PAPER-FAITHFUL metric: `best_subspan_em` on the first line
  of the answer (matches `evaluate_qa_responses.py`).
- `em_full`       — a LENIENT variant: `best_subspan_em` over the whole answer.
  Useful because modern chat models often add a Markdown heading / preamble on the
  first line, which can deflate the first-line metric even when the answer is right.

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
import statistics

from xopen import xopen

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


def main():
    logging.basicConfig(format="%(asctime)s - %(module)s - %(levelname)s - %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-documents", type=int, required=True)
    parser.add_argument("--gold-indices", type=int, nargs="+", required=True)
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args()

    slug = safe_model_slug(args.model)
    # Results are organized per-model: results/<slug>/...
    outdir = os.path.join(args.outdir, slug)
    os.makedirs(outdir, exist_ok=True)
    base = f"{outdir}/qa_{args.num_documents}docs_{slug}"

    per_example_rows = []
    aggregate = []

    for gold_index in args.gold_indices:
        path = find_prediction_file(args.num_documents, gold_index, slug)
        first_line_scores, full_scores = [], []
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
                first_line_scores.append(em_first)
                full_scores.append(em_full)
                per_example_rows.append(
                    {
                        "model": args.model,
                        "num_documents": args.num_documents,
                        "gold_index": gold_index,
                        "question": ex.get("question", ""),
                        "gold_answers": "|".join(ex.get("answers", [])),
                        "em_first_line": em_first,
                        "em_full": em_full,
                        "model_answer_failed": failed,
                        "model_answer": answer.replace("\n", " ").strip(),
                    }
                )
        aggregate.append(
            {
                "model": args.model,
                "num_documents": args.num_documents,
                "gold_index": gold_index,
                "n_examples": len(first_line_scores),
                "n_failed": n_failed,
                "best_subspan_em_first_line": statistics.mean(first_line_scores) if first_line_scores else float("nan"),
                "best_subspan_em_full": statistics.mean(full_scores) if full_scores else float("nan"),
            }
        )
        logger.info(
            "gold=%d n=%d  first_line=%.4f  full=%.4f  (%d failed)",
            gold_index,
            aggregate[-1]["n_examples"],
            aggregate[-1]["best_subspan_em_first_line"],
            aggregate[-1]["best_subspan_em_full"],
            n_failed,
        )

    # Write per-example CSV
    per_example_path = f"{base}_per_example.csv"
    with open(per_example_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_example_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_example_rows)

    # Write aggregate CSV
    summary_csv_path = f"{base}_summary.csv"
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregate[0].keys()))
        writer.writeheader()
        writer.writerows(aggregate)

    # Write aggregate JSON
    summary_json_path = f"{base}_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {"model": args.model, "task": "multi_document_qa", "num_documents": args.num_documents, "results": aggregate},
            f,
            indent=2,
        )

    logger.info("Wrote:\n  %s\n  %s\n  %s", per_example_path, summary_csv_path, summary_json_path)


if __name__ == "__main__":
    main()
