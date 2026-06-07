#!/usr/bin/env python3
"""Given a data file with questions and retrieval results, query a hosted API model
(OpenAI, Anthropic, Google, or any OpenAI-compatible provider) for responses.

This is the API-based counterpart to the local-GPU scripts
(`get_qa_responses_from_{mpt,longchat,llama_2}.py`). It produces output in the
exact same JSONL format, so `evaluate_qa_responses.py` works unchanged.

Examples:

    python -u ./scripts/get_qa_responses_from_api.py \
        --input-path qa_data/20_total_documents/nq-open-20_total_documents_gold_at_0.jsonl.gz \
        --model gpt-5.1 \
        --max-examples 100 \
        --output-path qa_predictions/.../gpt-5.1-predictions.jsonl.gz

The retrieval results are used in the exact order that they're given.
"""
import argparse
import dataclasses
import json
import logging
import pathlib
import random
import sys
from copy import deepcopy

from tqdm import tqdm
from xopen import xopen

from lost_in_the_middle.api_models import generate_responses, uses_sampling_params
from lost_in_the_middle.prompting import (
    Document,
    get_closedbook_qa_prompt,
    get_qa_prompt,
)

logger = logging.getLogger(__name__)
random.seed(0)


def main(
    input_path,
    model_name,
    temperature,
    top_p,
    closedbook,
    prompt_mention_random_ordering,
    use_random_ordering,
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
    all_model_documents = []

    # Fetch all of the prompts
    with xopen(input_path) as fin:
        for line in tqdm(fin):
            input_example = json.loads(line)
            # Get the prediction for the input example
            question = input_example["question"]
            if closedbook:
                documents = []
            else:
                documents = []
                for ctx in deepcopy(input_example["ctxs"]):
                    documents.append(Document.from_dict(ctx))
                if not documents:
                    raise ValueError(f"Did not find any documents for example: {input_example}")

            if use_random_ordering:
                # Randomly order only the distractors (isgold is False), keeping isgold documents
                # at their existing index.
                (original_gold_index,) = [idx for idx, doc in enumerate(documents) if doc.isgold is True]
                original_gold_document = documents[original_gold_index]
                distractors = [doc for doc in documents if doc.isgold is False]
                random.shuffle(distractors)
                distractors.insert(original_gold_index, original_gold_document)
                documents = distractors

            if closedbook:
                prompt = get_closedbook_qa_prompt(question)
            else:
                prompt = get_qa_prompt(
                    question,
                    documents,
                    mention_random_ordering=prompt_mention_random_ordering,
                    query_aware_contextualization=query_aware_contextualization,
                )

            prompts.append(prompt)
            examples.append(deepcopy(input_example))
            all_model_documents.append(documents)

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
        for example, model_documents, prompt, response, failed in zip(
            examples, all_model_documents, prompts, responses, failures
        ):
            output_example = deepcopy(example)
            # Add some extra metadata to the output example
            output_example["model_prompt"] = prompt
            output_example["model_documents"] = [dataclasses.asdict(document) for document in model_documents]
            output_example["model_answer"] = response
            output_example["model_answer_failed"] = failed
            output_example["model"] = model_name
            output_example["model_temperature"] = recorded_temperature
            output_example["model_top_p"] = recorded_top_p
            output_example["model_prompt_mention_random_ordering"] = prompt_mention_random_ordering
            output_example["model_use_random_ordering"] = use_random_ordering
            f.write(json.dumps(output_example) + "\n")


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(module)s - %(levelname)s - %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", help="Path to data with questions and documents to use.", required=True)
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
    parser.add_argument(
        "--closedbook", action="store_true", help="Run the model in closed-book mode (i.e., don't use documents)."
    )
    parser.add_argument(
        "--prompt-mention-random-ordering",
        action="store_true",
        help="Mention that search results are ordered randomly in the prompt",
    )
    parser.add_argument(
        "--use-random-ordering",
        action="store_true",
        help="Randomize the ordering of the distractors, rather than sorting by relevance.",
    )
    parser.add_argument(
        "--query-aware-contextualization",
        action="store_true",
        help="Place the question both before and after the documents.",
    )
    parser.add_argument("--output-path", help="Path to write output file of generated responses", required=True)
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
        args.closedbook,
        args.prompt_mention_random_ordering,
        args.use_random_ordering,
        args.query_aware_contextualization,
        args.max_new_tokens,
        args.max_examples,
        args.num_workers,
        args.reasoning_effort,
        args.reasoning_buffer,
        args.output_path,
    )
    logger.info("finished running %s", sys.argv[0])
