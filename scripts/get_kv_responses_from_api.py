#!/usr/bin/env python3
"""Given a data file with KV records, query a hosted API model (OpenAI, Anthropic,
Google, or any OpenAI-compatible provider) for key-value retrieval results.

This is the API-based counterpart to the local-GPU scripts
(`get_kv_responses_from_{mpt,longchat}.py`). It produces output in the exact same
JSONL format, so `evaluate_kv_responses.py` works unchanged.

Example:

    python -u ./scripts/get_kv_responses_from_api.py \
        --input-path kv_retrieval_data/kv-retrieval-140_keys.jsonl.gz \
        --gold-index 0 \
        --model gpt-5.1 \
        --max-examples 100 \
        --output-path kv_predictions/.../gpt-5.1-predictions.jsonl.gz

The KV records are used in the exact order that they're given.
"""
import argparse
import json
import logging
import pathlib
import random
import sys
from copy import deepcopy

from tqdm import tqdm
from xopen import xopen

from lost_in_the_middle.api_models import generate_responses, uses_sampling_params
from lost_in_the_middle.prompting import get_kv_retrieval_prompt

logger = logging.getLogger(__name__)
random.seed(0)


def main(
    input_path,
    model_name,
    temperature,
    top_p,
    gold_index,
    query_aware_contextualization,
    max_new_tokens,
    max_examples,
    num_workers,
    reasoning_effort,
    reasoning_buffer,
    output_path,
):
    # Create directory for output path if it doesn't exist.
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    examples = []
    prompts = []
    all_model_ordered_kv_records = []

    # Fetch all of the prompts
    with xopen(input_path) as fin:
        for line in tqdm(fin):
            input_example = json.loads(line)
            ordered_kv_records = deepcopy(input_example["ordered_kv_records"])
            key = input_example["key"]
            value = input_example["value"]
            original_kv_index = ordered_kv_records.index([key, value])
            # Remove the kv to retrieve from its original index
            original_kv = ordered_kv_records.pop(original_kv_index)
            # Insert it at the specified gold index
            ordered_kv_records.insert(gold_index, original_kv)

            kv_prompt = get_kv_retrieval_prompt(
                data=ordered_kv_records, key=key, query_aware_contextualization=query_aware_contextualization
            )

            prompts.append(kv_prompt)
            examples.append(deepcopy(input_example))
            all_model_ordered_kv_records.append(ordered_kv_records)

            if max_examples is not None and len(prompts) >= max_examples:
                logger.info(f"Reached the requested sample size of {max_examples} examples; stopping read.")
                break

    logger.info(f"Loaded {len(prompts)} prompts to process")

    responses, failures = generate_responses(
        model_name=model_name,
        prompts=prompts,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        num_workers=num_workers,
        reasoning_effort=reasoning_effort,
        reasoning_buffer=reasoning_buffer,
        return_failures=True,
    )

    # Reasoning models (e.g. gpt-5.x / o-series) ignore temperature/top_p, so
    # recording them as if they were applied would be misleading. Record None in
    # that case so the output faithfully reflects the request.
    sampling_applied = uses_sampling_params(model_name)
    recorded_temperature = temperature if sampling_applied else None
    recorded_top_p = top_p if sampling_applied else None

    with xopen(output_path, "w") as f:
        for example, ordered_kv_records, prompt, response, failed in zip(
            examples, all_model_ordered_kv_records, prompts, responses, failures
        ):
            output_example = deepcopy(example)
            # Add some extra metadata to the output example
            output_example["model_prompt"] = prompt
            output_example["model_answer"] = response
            output_example["model_answer_failed"] = failed
            output_example["model"] = model_name
            output_example["model_temperature"] = recorded_temperature
            output_example["model_top_p"] = recorded_top_p
            output_example["model_ordered_kv_records"] = ordered_kv_records
            f.write(json.dumps(output_example) + "\n")


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(module)s - %(levelname)s - %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", help="Path to data with KV records to use.", required=True)
    parser.add_argument(
        "--model",
        help=(
            "Model to use in generating responses, e.g. 'gpt-5.1', 'claude-opus-4-5', 'gemini-2.5-pro'. "
            "Prefix with a provider to be explicit, e.g. 'deepseek:deepseek-chat'."
        ),
        required=True,
    )
    parser.add_argument("--temperature", help="Temperature to use in generation", type=float, default=0.0)
    parser.add_argument("--top-p", help="Top-p to use in generation", type=float, default=1.0)
    parser.add_argument("--output-path", help="Path to write output file of generated responses", required=True)
    parser.add_argument("--gold-index", help="Move the key to retrieve to this index", type=int, required=True)
    parser.add_argument(
        "--query-aware-contextualization", action="store_true", help="Use query-aware contextualization"
    )
    parser.add_argument(
        "--max-new-tokens",
        help="Maximum number of new tokens to generate",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--max-examples",
        help="If set, only run on the first N examples of the input file (useful for cheap sampling).",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--num-workers",
        help="Number of concurrent API requests. Lower this if you hit rate limits.",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["minimal", "low", "medium", "high"],
        default="low",
        help=(
            "Reasoning effort for reasoning models (gpt-5.x / o-series). This task wants a "
            "short answer, so 'low' keeps hidden-token spend (and the chance of truncating the "
            "answer) down. Ignored by non-reasoning models. See REASONING_TOKENS.md."
        ),
    )
    parser.add_argument(
        "--reasoning-buffer",
        type=int,
        default=4000,
        help=(
            "Extra completion-token budget granted to reasoning models on top of --max-new-tokens, "
            "to cover hidden reasoning before the visible answer. Raise it if you see "
            "'empty content with finish_reason=length' warnings. See REASONING_TOKENS.md."
        ),
    )
    args = parser.parse_args()

    logger.info("running %s", " ".join(sys.argv))
    main(
        args.input_path,
        args.model,
        args.temperature,
        args.top_p,
        args.gold_index,
        args.query_aware_contextualization,
        args.max_new_tokens,
        args.max_examples,
        args.num_workers,
        args.reasoning_effort,
        args.reasoning_buffer,
        args.output_path,
    )
    logger.info("finished running %s", sys.argv[0])
