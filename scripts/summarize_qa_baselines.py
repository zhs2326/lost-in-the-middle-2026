#!/usr/bin/env python3
"""Summarize the QA *baseline* prediction files (oracle + closed-book) for one model.

These two runs are reference points for the position-sweep curves:
- **oracle**     — gold document only -> the accuracy *ceiling*.
- **closedbook** — no documents at all -> the parametric-memory *floor* (how much
  the model already "knows" without any context).

Reads the prediction JSONL(.gz) written by ``get_qa_responses_from_api.py`` for
each condition and writes, to ``<outdir>/<slug>/``:

- ``qa_baselines_<slug>_per_example.csv`` — one row per question/condition,
- ``qa_baselines_<slug>_summary.{csv,json}`` — one row per condition.

Both metrics are recorded, exactly as in ``summarize_qa_results.py``:
``best_subspan_em`` on the first line (paper-faithful) and over the whole answer
(lenient). ``plot_qa_comparison.py`` reads the summary CSV to draw the reference
lines.

Example:

    python ./scripts/summarize_qa_baselines.py --model gemini-2.5-flash --outdir results
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

# condition -> directory + filename stem the API script writes.
CONDITIONS = {
    "oracle": ("qa_predictions/oracle", "nq-open-oracle"),
    "closedbook": ("qa_predictions/closedbook", "nq-open-closedbook"),
}


def safe_model_slug(model_name):
    return model_name.replace("/", "_").replace(":", "_")


def find_prediction_file(directory, stem, slug):
    pattern = f"{directory}/{stem}-{slug}-predictions.jsonl.gz"
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No prediction file matched: {pattern}")
    return sorted(matches)[0]


def main():
    logging.basicConfig(format="%(asctime)s - %(module)s - %(levelname)s - %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args()

    slug = safe_model_slug(args.model)
    outdir = os.path.join(args.outdir, slug)
    os.makedirs(outdir, exist_ok=True)
    base = f"{outdir}/qa_baselines_{slug}"

    per_example_rows = []
    aggregate = []

    for condition, (directory, stem) in CONDITIONS.items():
        path = find_prediction_file(directory, stem, slug)
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
                        "condition": condition,
                        "question": ex.get("question", ""),
                        "gold_answers": "|".join(ex.get("answers", [])),
                        "em_first_line": em_first,
                        "em_full": em_full,
                        "model_answer": answer.replace("\n", " ").strip(),
                    }
                )
        aggregate.append(
            {
                "condition": condition,
                "n": len(first_line_scores),
                "n_failed": n_failed,
                "best_subspan_em_first_line": statistics.mean(first_line_scores) if first_line_scores else float("nan"),
                "best_subspan_em_full": statistics.mean(full_scores) if full_scores else float("nan"),
            }
        )
        logger.info(
            "%s: n=%d  first_line=%.4f  full=%.4f  (%d failed)",
            condition,
            aggregate[-1]["n"],
            aggregate[-1]["best_subspan_em_first_line"],
            aggregate[-1]["best_subspan_em_full"],
            n_failed,
        )

    per_example_path = f"{base}_per_example.csv"
    with open(per_example_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_example_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_example_rows)

    summary_csv_path = f"{base}_summary.csv"
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregate[0].keys()))
        writer.writeheader()
        writer.writerows(aggregate)

    summary_json_path = f"{base}_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "task": "qa_baselines", "results": aggregate}, f, indent=2)

    logger.info("Wrote:\n  %s\n  %s\n  %s", per_example_path, summary_csv_path, summary_json_path)


if __name__ == "__main__":
    main()
