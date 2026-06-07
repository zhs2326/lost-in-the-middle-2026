#!/usr/bin/env python3
"""Convenience runner: sweep an API model across all gold positions for the
multi-document QA and/or key-value retrieval tasks, evaluate each run, and print
a summary table tracing the "lost in the middle" curve.

Cross-platform (works on Windows PowerShell, no bash `for` loops needed).

Example:

    python ./scripts/run_api_sweep.py --model gpt-5.1 --max-examples 100
    python ./scripts/run_api_sweep.py --model claude-opus-4-5 --task qa --max-examples 50
    python ./scripts/run_api_sweep.py --model gemini-2.5-pro --num-workers 8

Requires the relevant API key in the environment (OPENAI_API_KEY /
ANTHROPIC_API_KEY / GEMINI_API_KEY / ...). See EXPERIMENTS.md.
"""
import argparse
import json
import logging
import pathlib
import statistics
import sys

from xopen import xopen

from lost_in_the_middle.api_models import detect_provider, generate_responses
from lost_in_the_middle.metrics import best_subspan_em
from lost_in_the_middle.prompting import Document, get_kv_retrieval_prompt, get_qa_prompt

logger = logging.getLogger(__name__)

# Gold positions used in the paper for the 20-document QA and 140-key KV settings.
QA_GOLD_INDICES = [0, 4, 9, 14, 19]
KV_GOLD_INDICES = [0, 34, 69, 104, 139]
QA_INPUT_TEMPLATE = "qa_data/20_total_documents/nq-open-20_total_documents_gold_at_{gold_index}.jsonl.gz"
KV_INPUT_PATH = "kv_retrieval_data/kv-retrieval-140_keys.jsonl.gz"


def safe_model_slug(model_name):
    return model_name.replace("/", "_").replace(":", "_")


def load_qa_prompts(input_path, max_examples):
    examples, prompts = [], []
    with xopen(input_path) as fin:
        for line in fin:
            ex = json.loads(line)
            docs = [Document.from_dict(c) for c in ex["ctxs"]]
            prompts.append(get_qa_prompt(ex["question"], docs, mention_random_ordering=False,
                                         query_aware_contextualization=False))
            examples.append(ex)
            if max_examples is not None and len(prompts) >= max_examples:
                break
    return examples, prompts


def load_kv_prompts(input_path, gold_index, max_examples):
    examples, prompts = [], []
    with xopen(input_path) as fin:
        for line in fin:
            ex = json.loads(line)
            records = list(ex["ordered_kv_records"])
            kv = records.pop(records.index([ex["key"], ex["value"]]))
            records.insert(gold_index, kv)
            prompts.append(get_kv_retrieval_prompt(data=records, key=ex["key"]))
            examples.append(ex)
            if max_examples is not None and len(prompts) >= max_examples:
                break
    return examples, prompts


def score_qa(examples, responses):
    scores = []
    for ex, resp in zip(examples, responses):
        prediction = resp.split("\n")[0].strip()
        scores.append(best_subspan_em(prediction=prediction, ground_truths=ex["answers"]))
    return statistics.mean(scores) if scores else float("nan")


def score_kv(examples, responses):
    scores = []
    for ex, resp in zip(examples, responses):
        scores.append(1.0 if ex["value"].lower() in resp.lower() else 0.0)
    return statistics.mean(scores) if scores else float("nan")


def write_predictions(output_path, examples, prompts, responses, model_name, failures=None, extra_per_example=None):
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if failures is None:
        failures = [False] * len(responses)
    with xopen(output_path, "w") as f:
        for idx, (ex, prompt, resp, failed) in enumerate(zip(examples, prompts, responses, failures)):
            out = dict(ex)
            out["model_prompt"] = prompt
            out["model_answer"] = resp
            out["model_answer_failed"] = failed
            out["model"] = model_name
            if extra_per_example:
                out.update(extra_per_example(idx, ex))
            f.write(json.dumps(out) + "\n")


def run_qa(model_name, max_examples, num_workers, max_new_tokens, reasoning_effort, reasoning_buffer):
    results = {}
    for gold_index in QA_GOLD_INDICES:
        input_path = QA_INPUT_TEMPLATE.format(gold_index=gold_index)
        examples, prompts = load_qa_prompts(input_path, max_examples)
        logger.info("[QA gold=%d] generating %d responses ...", gold_index, len(prompts))
        responses, failures = generate_responses(
            model_name, prompts, max_new_tokens=max_new_tokens, num_workers=num_workers,
            reasoning_effort=reasoning_effort, reasoning_buffer=reasoning_buffer, return_failures=True,
        )
        out_path = f"qa_predictions/20_total_documents/gold_at_{gold_index}-{safe_model_slug(model_name)}-predictions.jsonl.gz"
        write_predictions(out_path, examples, prompts, responses, model_name, failures=failures)
        results[gold_index] = score_qa(examples, responses)
        logger.info(
            "[QA gold=%d] best_subspan_em=%.4f (%d/%d failed)",
            gold_index, results[gold_index], sum(failures), len(failures),
        )
    return results


def run_kv(model_name, max_examples, num_workers, max_new_tokens, reasoning_effort, reasoning_buffer):
    results = {}
    for gold_index in KV_GOLD_INDICES:
        examples, prompts = load_kv_prompts(KV_INPUT_PATH, gold_index, max_examples)
        logger.info("[KV gold=%d] generating %d responses ...", gold_index, len(prompts))
        responses, failures = generate_responses(
            model_name, prompts, max_new_tokens=max_new_tokens, num_workers=num_workers,
            reasoning_effort=reasoning_effort, reasoning_buffer=reasoning_buffer, return_failures=True,
        )
        out_path = f"kv_predictions/kv-retrieval-140_keys_gold_at_{gold_index}-{safe_model_slug(model_name)}-predictions.jsonl.gz"
        write_predictions(out_path, examples, prompts, responses, model_name, failures=failures)
        results[gold_index] = score_kv(examples, responses)
        logger.info(
            "[KV gold=%d] accuracy=%.4f (%d/%d failed)",
            gold_index, results[gold_index], sum(failures), len(failures),
        )
    return results


def print_table(title, metric_name, results):
    print(f"\n=== {title} ===")
    print(f"{'gold_index':>12} | {metric_name}")
    print("-" * 34)
    for gold_index, value in results.items():
        bar = "#" * int(round(value * 20))
        print(f"{gold_index:>12} | {value:.4f}  {bar}")


def main():
    logging.basicConfig(format="%(asctime)s - %(module)s - %(levelname)s - %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="e.g. gpt-5.1, claude-opus-4-5, gemini-2.5-pro")
    parser.add_argument("--task", choices=["qa", "kv", "both"], default="both")
    parser.add_argument("--max-examples", type=int, default=100,
                        help="Examples per gold position (default 100). Use None-equivalent 0 for the full set.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument(
        "--reasoning-effort",
        choices=["minimal", "low", "medium", "high"],
        default="low",
        help="Reasoning effort for reasoning models (gpt-5.x / o-series). Ignored otherwise. "
        "See REASONING_TOKENS.md.",
    )
    parser.add_argument(
        "--reasoning-buffer",
        type=int,
        default=4000,
        help="Extra completion-token budget for hidden reasoning, on top of --max-new-tokens. "
        "Raise it if you see 'finish_reason=length' warnings. See REASONING_TOKENS.md.",
    )
    args = parser.parse_args()

    max_examples = None if args.max_examples in (0, None) else args.max_examples
    logger.info("Model '%s' resolved to provider '%s'", args.model, detect_provider(args.model))

    summary = {}
    if args.task in ("qa", "both"):
        summary["qa"] = run_qa(
            args.model, max_examples, args.num_workers, args.max_new_tokens,
            args.reasoning_effort, args.reasoning_buffer,
        )
    if args.task in ("kv", "both"):
        summary["kv"] = run_kv(
            args.model, max_examples, args.num_workers, args.max_new_tokens,
            args.reasoning_effort, args.reasoning_buffer,
        )

    print(f"\n########## SUMMARY for {args.model} (max_examples={max_examples}) ##########")
    if "qa" in summary:
        print_table("Multi-document QA (20 docs)", "best_subspan_em", summary["qa"])
    if "kv" in summary:
        print_table("Key-value retrieval (140 keys)", "accuracy", summary["kv"])
    print("\nLower values for middle gold positions vs. the ends = the 'lost in the middle' effect.")


if __name__ == "__main__":
    logger.info("running %s", " ".join(sys.argv))
    main()
