#!/usr/bin/env python3
"""Run multi-document QA prompts through Gemini's **Batch API** (50% cheaper, async).

This is the batch-mode counterpart to ``get_qa_responses_from_api.py``. It exists
for one concrete reason (see findings/EXPERIMENT_LOG.md, the gemini long-context
tier): the synchronous path hits the per-minute TPM/RPM quota wall once prompts
get large (200+ docs, ~33K tokens each), forcing workers=1 or failing 8-11%/pos.
Batch mode sidesteps that — it runs against a separate, far larger async quota,
completes within a 24h window (usually minutes), and is billed at **50% of the
standard token price**.

It builds prompts with the *exact same* ``lost_in_the_middle.prompting`` code as
the sync script and writes output in the *exact same* JSONL schema (adding
``model_answer_failed``), so ``summarize_qa_results.py`` / ``evaluate_qa_responses.py``
work on the result unchanged.

One batch job covers all the input position-files at once. Each request carries a
``key`` of the form ``in<input_idx>-ex<example_idx>`` so responses route back to the
right position file + example, regardless of the order the batch returns them.

Modes:
  --mode run     (default) submit the job, poll to completion, then write outputs.
  --mode submit  build + upload + create the job, save a state file, and exit.
  --mode fetch   resume from a state file: poll (if needed) and write outputs.

The state file (``--state-file``, default next to the outputs) holds the job name,
uploaded-file name, and the run parameters — enough to re-derive the manifest
deterministically (prompts are a pure function of the inputs) and recover if the
process dies mid-poll.

Example (the 200-doc gemini sweep, n=100/position):

    python ./scripts/get_qa_responses_from_gemini_batch.py \
        --input-glob "qa_data/200_total_documents/nq-open-200_total_documents_gold_at_*.jsonl.gz" \
        --model gemini-2.5-flash --max-examples 100 --reasoning-buffer 0 \
        --output-dir qa_predictions/200_total_documents
"""
import argparse
import dataclasses
import glob as globmod
import io
import json
import logging
import os
import pathlib
import sys
import time
from copy import deepcopy

from xopen import xopen

from lost_in_the_middle.prompting import Document, get_closedbook_qa_prompt, get_qa_prompt

# Load API keys from .env exactly like api_models.py does on import — otherwise a
# bare genai.Client() finds no key. No-op if python-dotenv / .env are absent, and
# it never overrides a key already in the real environment.
try:
    from dotenv import load_dotenv

    load_dotenv(override=False)
except ImportError:
    pass

logger = logging.getLogger(__name__)

# Batch job terminal states (Gemini Developer API). SUCCEEDED is the only success.
_TERMINAL_OK = "JOB_STATE_SUCCEEDED"
_TERMINAL_BAD = {"JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}


def safe_model_slug(model_name):
    return model_name.replace("/", "_").replace(":", "_")


def _strip_gz_jsonl(name):
    for suffix in (".jsonl.gz", ".jsonl", ".gz"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def output_path_for(input_path, output_dir, slug):
    """Mirror the sync script's naming: <stem>-<slug>-predictions.jsonl.gz."""
    stem = _strip_gz_jsonl(os.path.basename(input_path))
    return os.path.join(output_dir, f"{stem}-{slug}-predictions.jsonl.gz")


def build_manifest(input_paths, closedbook, max_examples, prompt_mention_random_ordering,
                   query_aware_contextualization):
    """Re-derive (deterministically) every request: its key, prompt, documents, and
    the source example. Same prompt-construction as get_qa_responses_from_api.py."""
    manifest = []
    for input_idx, input_path in enumerate(input_paths):
        n_read = 0
        with xopen(input_path) as fin:
            for line in fin:
                input_example = json.loads(line)
                question = input_example["question"]
                if closedbook:
                    documents = []
                else:
                    documents = [Document.from_dict(ctx) for ctx in deepcopy(input_example["ctxs"])]
                    if not documents:
                        raise ValueError(f"Did not find any documents for example: {input_example}")

                if closedbook:
                    prompt = get_closedbook_qa_prompt(question)
                else:
                    prompt = get_qa_prompt(
                        question, documents,
                        mention_random_ordering=prompt_mention_random_ordering,
                        query_aware_contextualization=query_aware_contextualization,
                    )

                manifest.append({
                    "key": f"in{input_idx}-ex{n_read}",
                    "input_idx": input_idx,
                    "input_path": input_path,
                    "example_idx": n_read,
                    "example": deepcopy(input_example),
                    "documents": documents,
                    "prompt": prompt,
                })
                n_read += 1
                if max_examples is not None and n_read >= max_examples:
                    break
    return manifest


def write_batch_input_jsonl(manifest, jsonl_path, temperature, top_p, max_new_tokens, reasoning_buffer):
    """Write the batch input file: one {key, request} line per prompt.

    The request's generationConfig is serialized by the SDK so it's the canonical
    camelCase REST form (topP / maxOutputTokens / thinkingConfig.thinkingBudget) —
    no hand-rolled casing to get wrong. thinking_budget=0 disables Gemini 2.5
    thinking, matching the sync runs (--reasoning-buffer 0)."""
    from google.genai import types

    config_kwargs = {"temperature": temperature, "top_p": top_p, "max_output_tokens": max_new_tokens}
    # Gemini 2.5 thinks by default and bills thinking against max_output_tokens.
    # Grant answer budget + buffer and cap thinking; buffer 0 => thinking off.
    config_kwargs["max_output_tokens"] = max_new_tokens + reasoning_buffer
    config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=reasoning_buffer)
    gen_config = types.GenerateContentConfig(**config_kwargs).model_dump(
        by_alias=True, exclude_none=True, mode="json"
    )

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in manifest:
            line = {
                "key": rec["key"],
                "request": {
                    "contents": [{"role": "user", "parts": [{"text": rec["prompt"]}]}],
                    "generationConfig": gen_config,
                },
            }
            f.write(json.dumps(line) + "\n")
    return gen_config


def _extract_text(response_dict):
    """Pull visible text out of a GenerateContentResponse dict, defensively.

    Mirrors api_models._generate_google: a deterministic empty (safety block /
    MAX_TOKENS) returns "" rather than raising. Returns (text, finish_reason)."""
    candidates = response_dict.get("candidates") or []
    for cand in candidates:
        parts = ((cand.get("content") or {}).get("parts")) or []
        text = "".join((p.get("text") or "") for p in parts).strip()
        if text:
            return text, cand.get("finishReason")
    finish = candidates[0].get("finishReason") if candidates else None
    return "", finish


def submit(client, manifest, args, slug, state_path):
    from google.genai import types

    work_dir = pathlib.Path(state_path).parent
    work_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = str(work_dir / f"_batch_input_{slug}.jsonl")

    gen_config = write_batch_input_jsonl(
        manifest, jsonl_path, args.temperature, args.top_p, args.max_new_tokens, args.reasoning_buffer
    )
    size_mb = os.path.getsize(jsonl_path) / 1e6
    logger.info("Wrote %d requests to %s (%.1f MB)", len(manifest), jsonl_path, size_mb)

    uploaded = client.files.upload(
        file=jsonl_path,
        config=types.UploadFileConfig(display_name=f"qa-batch-{slug}", mime_type="jsonl"),
    )
    logger.info("Uploaded input file: %s", uploaded.name)

    job = client.batches.create(
        model=_strip_provider_prefix(args.model),
        src=uploaded.name,
        config={"display_name": f"qa-{slug}-{len(manifest)}reqs"},
    )
    logger.info("Created batch job: %s (state=%s)", job.name, getattr(job.state, "name", job.state))

    state = {
        "job_name": job.name,
        "uploaded_file": uploaded.name,
        "input_jsonl": jsonl_path,
        "model": args.model,
        "slug": slug,
        "output_dir": args.output_dir,
        "input_paths": [rec_path for rec_path in dict.fromkeys(r["input_path"] for r in manifest)],
        "closedbook": args.closedbook,
        "max_examples": args.max_examples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "reasoning_buffer": args.reasoning_buffer,
        "prompt_mention_random_ordering": args.prompt_mention_random_ordering,
        "query_aware_contextualization": args.query_aware_contextualization,
    }
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    logger.info("Saved state to %s", state_path)
    return job.name


def poll(client, job_name, poll_interval):
    while True:
        job = client.batches.get(name=job_name)
        state = getattr(job.state, "name", str(job.state))
        logger.info("Batch %s state=%s", job_name, state)
        if state == _TERMINAL_OK:
            return job
        if state in _TERMINAL_BAD:
            raise RuntimeError(f"Batch job {job_name} ended in {state}: {getattr(job, 'error', None)}")
        time.sleep(poll_interval)


def _iter_result_lines(client, job):
    """Yield parsed result dicts from a finished batch job, whether the destination
    is a downloadable file or inlined responses."""
    dest = job.dest
    file_name = getattr(dest, "file_name", None)
    if file_name:
        raw = client.files.download(file=file_name)
        text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
        for line in io.StringIO(text):
            line = line.strip()
            if line:
                yield json.loads(line)
        return
    # Fallback: inlined responses (small jobs). Normalize to the same {key,response} shape.
    for item in (getattr(dest, "inlined_responses", None) or []):
        resp = getattr(item, "response", None)
        err = getattr(item, "error", None)
        key = (getattr(item, "metadata", None) or {}).get("key")
        out = {"key": key}
        if resp is not None:
            out["response"] = resp.model_dump(by_alias=True, mode="json") if hasattr(resp, "model_dump") else resp
        if err is not None:
            out["error"] = err
        yield out


def write_outputs(client, job, manifest, args, slug):
    by_key = {rec["key"]: rec for rec in manifest}
    responses, errors = {}, {}
    for result in _iter_result_lines(client, job):
        key = result.get("key")
        if key is None:
            continue
        if "response" in result and result["response"] is not None:
            text, _finish = _extract_text(result["response"])
            responses[key] = text
        else:
            errors[key] = result.get("error")
            responses[key] = ""  # failed -> empty, flagged below

    missing = [k for k in by_key if k not in responses]
    if missing:
        logger.warning("%d/%d requests had NO result line in the batch output", len(missing), len(by_key))

    recorded_temperature = args.temperature
    recorded_top_p = args.top_p

    # Group records by their destination output file, preserving example order.
    groups = {}
    for rec in manifest:
        out_path = output_path_for(rec["input_path"], args.output_dir, slug)
        groups.setdefault(out_path, []).append(rec)

    total_failed = 0
    for out_path, recs in groups.items():
        pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        recs.sort(key=lambda r: r["example_idx"])
        with xopen(out_path, "w") as f:
            for rec in recs:
                key = rec["key"]
                failed = (key in errors) or (key in missing)
                total_failed += int(failed)
                output_example = deepcopy(rec["example"])
                output_example["model_prompt"] = rec["prompt"]
                output_example["model_documents"] = [dataclasses.asdict(d) for d in rec["documents"]]
                output_example["model_answer"] = responses.get(key, "")
                output_example["model_answer_failed"] = failed
                output_example["model"] = _strip_provider_prefix(args.model)
                output_example["model_temperature"] = recorded_temperature
                output_example["model_top_p"] = recorded_top_p
                output_example["model_prompt_mention_random_ordering"] = args.prompt_mention_random_ordering
                output_example["model_use_random_ordering"] = False
                f.write(json.dumps(output_example) + "\n")
        logger.info("Wrote %d predictions -> %s", len(recs), out_path)

    logger.info("DONE: %d outputs across %d files; %d failed requests",
                len(manifest), len(groups), total_failed)
    if errors:
        sample = list(errors.items())[:3]
        logger.warning("Sample errors: %s", sample)
    return total_failed


def _strip_provider_prefix(model_name):
    return model_name.split(":", 1)[1] if ":" in model_name else model_name


def resolve_inputs(args):
    inputs = list(args.input_path or [])
    if args.input_glob:
        inputs.extend(sorted(globmod.glob(args.input_glob)))
    # De-dup while preserving order.
    seen, ordered = set(), []
    for p in inputs:
        if p not in seen:
            seen.add(p)
            ordered.append(p)
    return ordered


def main():
    logging.basicConfig(format="%(asctime)s - %(module)s - %(levelname)s - %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-path", action="append", help="Input file (repeatable).")
    parser.add_argument("--input-glob", help="Glob of input files (e.g. '.../gold_at_*.jsonl.gz').")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--output-dir", required=True, help="Directory for per-position prediction files.")
    parser.add_argument("--closedbook", action="store_true")
    parser.add_argument("--prompt-mention-random-ordering", action="store_true")
    parser.add_argument("--query-aware-contextualization", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--reasoning-buffer", type=int, default=0,
                        help="Gemini thinking budget. 0 = thinking OFF (matches the sync gemini runs).")
    parser.add_argument("--mode", choices=["run", "submit", "fetch"], default="run")
    parser.add_argument("--state-file", default=None, help="Where to persist/resume job state.")
    parser.add_argument("--poll-interval", type=int, default=30)
    args = parser.parse_args()

    if _strip_provider_prefix(args.model).lower().startswith("gemini") is False:
        logger.warning("This script targets Gemini batch; got model '%s'.", args.model)

    from google import genai
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key) if api_key else genai.Client()
    slug = safe_model_slug(args.model)
    state_path = args.state_file or os.path.join(args.output_dir, f"_batch_state_{slug}.json")

    logger.info("running %s", " ".join(sys.argv))

    if args.mode == "fetch":
        with open(state_path, encoding="utf-8") as f:
            state = json.load(f)
        # Rebuild the manifest deterministically from the recorded inputs/params.
        ns = argparse.Namespace(**{**vars(args), **{
            "output_dir": state["output_dir"], "closedbook": state["closedbook"],
            "max_examples": state["max_examples"], "temperature": state["temperature"],
            "top_p": state["top_p"], "max_new_tokens": state["max_new_tokens"],
            "reasoning_buffer": state["reasoning_buffer"], "model": state["model"],
            "prompt_mention_random_ordering": state["prompt_mention_random_ordering"],
            "query_aware_contextualization": state["query_aware_contextualization"],
        }})
        manifest = build_manifest(state["input_paths"], ns.closedbook, ns.max_examples,
                                  ns.prompt_mention_random_ordering, ns.query_aware_contextualization)
        job = poll(client, state["job_name"], args.poll_interval)
        write_outputs(client, job, manifest, ns, state["slug"])
        return

    input_paths = resolve_inputs(args)
    if not input_paths:
        parser.error("No inputs: pass --input-path and/or --input-glob.")
    logger.info("Building manifest from %d input file(s)", len(input_paths))
    manifest = build_manifest(input_paths, args.closedbook, args.max_examples,
                              args.prompt_mention_random_ordering, args.query_aware_contextualization)
    logger.info("Manifest: %d total requests", len(manifest))

    job_name = submit(client, manifest, args, slug, state_path)
    if args.mode == "submit":
        logger.info("Submitted. Resume with: --mode fetch --state-file %s", state_path)
        return

    job = poll(client, job_name, args.poll_interval)
    write_outputs(client, job, manifest, args, slug)
    logger.info("finished running %s", sys.argv[0])


if __name__ == "__main__":
    main()
