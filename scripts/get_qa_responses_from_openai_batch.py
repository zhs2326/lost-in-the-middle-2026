#!/usr/bin/env python3
"""Run multi-document QA prompts through the **OpenAI Batch API** (50% cheaper, async).

The batch-mode counterpart to ``get_qa_responses_from_api.py`` for OpenAI models
(gpt-4.1, gpt-4o, ...). Same motivation as the Gemini batch script: long-context
sweeps (100+ docs) are a lot of large requests, and batch mode runs them against a
separate async quota at **50% of standard token price**, completing within a 24h
window. Unlike Gemini's batch (paid-tier only), OpenAI's Batch API is available on
any standard paid account.

It builds prompts with the *exact same* ``lost_in_the_middle.prompting`` code as
the sync script and writes output in the *exact same* JSONL schema (adding
``model_answer_failed``), so ``summarize_qa_results.py`` / ``evaluate_qa_responses.py``
work on the result unchanged.

CHUNKING (the reason this isn't one giant job): OpenAI caps the **enqueued tokens**
a batch may hold at once (e.g. 1.35M for gpt-4.1 on a low tier). A 100-doc sweep is
~14K tokens × 1000 requests ≈ 16M tokens — far over the cap — so a single batch
fails validation with ``token_limit_exceeded``. We therefore pack requests into
chunks under ``--max-enqueued-tokens`` (estimated as chars/4 + max_tokens, which
overcounts vs. real tokenization, giving headroom) and submit them **sequentially**:
only one batch is ever in flight, so the live enqueued-token total stays under the
cap. Responses from all chunks are merged by ``custom_id`` and written once at the
end, so the per-position output files are identical to a single-job run.

One logical run covers all the input position-files. Each request carries a
``custom_id`` of the form ``in<input_idx>-ex<example_idx>`` so responses route back
to the right position file + example regardless of return order or chunk boundary.

NOTE: targets **non-reasoning** chat models (gpt-4.1 / gpt-4o), which take
temperature/top_p/max_tokens directly. Reasoning models (gpt-5.x / o-series) reject
sampling params and need max_completion_tokens + a reasoning buffer — use the sync
path for those.

Modes:
  --mode run     (default) chunk, submit+poll each chunk sequentially, then write.
  --mode fetch   resume from a state file: re-poll the recorded chunk jobs, write.

Example (the 100-doc gpt-4.1 sweep, n=100/position):

    python ./scripts/get_qa_responses_from_openai_batch.py \
        --input-glob "qa_data/100_total_documents/nq-open-100_total_documents_gold_at_*.jsonl.gz" \
        --model gpt-4.1 --max-examples 100 \
        --output-dir qa_predictions/100_total_documents
"""
import argparse
import dataclasses
import glob as globmod
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
# bare OpenAI() finds no key. No-op if python-dotenv / .env are absent, and it
# never overrides a key already in the real environment.
try:
    from dotenv import load_dotenv

    load_dotenv(override=False)
except ImportError:
    pass

logger = logging.getLogger(__name__)

# OpenAI batch terminal states. "completed" is the only success; the others are
# terminal failures we should stop polling on.
_TERMINAL_OK = "completed"
_TERMINAL_BAD = {"failed", "expired", "cancelled", "cancelling"}


def safe_model_slug(model_name):
    return model_name.replace("/", "_").replace(":", "_")


def _strip_provider_prefix(model_name):
    return model_name.split(":", 1)[1] if ":" in model_name else model_name


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
    """Re-derive (deterministically) every request: its custom_id, prompt, documents,
    and the source example. Same prompt construction as get_qa_responses_from_api.py."""
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
                    "custom_id": f"in{input_idx}-ex{n_read}",
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


def estimate_tokens(prompt, max_new_tokens):
    """Conservative enqueued-token estimate: chars/4 (overcounts vs. real BPE for
    English text) + the output budget. Used only to size chunks under the cap."""
    return len(prompt) // 4 + max_new_tokens


def chunk_manifest(manifest, max_enqueued_tokens, max_new_tokens):
    """Greedily pack records into chunks whose estimated enqueued tokens stay under
    the cap. A single request larger than the cap goes alone (and is flagged)."""
    chunks, current, current_tokens = [], [], 0
    for rec in manifest:
        t = estimate_tokens(rec["prompt"], max_new_tokens)
        if t > max_enqueued_tokens:
            logger.warning("Single request %s ~%d tok exceeds the per-batch cap %d; sending it alone.",
                           rec["custom_id"], t, max_enqueued_tokens)
        if current and current_tokens + t > max_enqueued_tokens:
            chunks.append(current)
            current, current_tokens = [], 0
        current.append(rec)
        current_tokens += t
    if current:
        chunks.append(current)
    return chunks


def write_batch_input_jsonl(chunk, jsonl_path, model, temperature, top_p, max_new_tokens):
    """Write one chunk's batch input file: one chat-completions request per line.

    Each line is {custom_id, method, url, body}, where body is a standard
    /v1/chat/completions request. gpt-4.1 is non-reasoning, so temperature/top_p/
    max_tokens are sent directly (matching the sync path's greedy default)."""
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in chunk:
            line = {
                "custom_id": rec["custom_id"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [{"role": "user", "content": rec["prompt"]}],
                    "max_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                },
            }
            f.write(json.dumps(line) + "\n")


def submit_chunk(client, chunk, model, args, slug, chunk_idx, work_dir):
    jsonl_path = str(work_dir / f"_batch_input_{slug}_chunk{chunk_idx:02d}.jsonl")
    write_batch_input_jsonl(chunk, jsonl_path, model, args.temperature, args.top_p, args.max_new_tokens)
    size_mb = os.path.getsize(jsonl_path) / 1e6
    est = sum(estimate_tokens(r["prompt"], args.max_new_tokens) for r in chunk)
    logger.info("Chunk %d: %d requests, %.1f MB, ~%d est. enqueued tokens", chunk_idx, len(chunk), size_mb, est)

    def _upload():
        with open(jsonl_path, "rb") as fh:
            return client.files.create(file=fh, purpose="batch")

    uploaded = _with_retry(_upload, f"upload chunk {chunk_idx}")
    job = _with_retry(lambda: client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": f"qa-{slug}-chunk{chunk_idx}"},
    ), f"create batch chunk {chunk_idx}")
    logger.info("Chunk %d: created batch %s (status=%s)", chunk_idx, job.id, job.status)
    # Clean up the local chunk input file; the upload now lives server-side.
    try:
        os.remove(jsonl_path)
    except OSError:
        pass
    return job.id, uploaded.id


def _with_retry(fn, what, retries=12):
    """Retry a network call through transient errors so a flaky connection never
    aborts a multi-chunk run. The batch jobs run server-side and complete reliably;
    it's only these poll/download HTTP calls that fail intermittently on a
    geo-restricted / VPN'd connection. Besides timeouts and 5xx, we treat the OpenAI
    403 ``unsupported_country_region_territory`` as transient: it depends on which
    egress route the request took, so a later attempt usually lands on a good one.
    Backoff is longer/capped higher to ride out a multi-minute block window. Batch
    *state* failures are not caught here — those surface from poll() as RuntimeError."""
    import openai

    transient = (
        openai.APITimeoutError, openai.APIConnectionError,
        openai.RateLimitError, openai.InternalServerError,
    )

    def _is_geo_403(e):
        return isinstance(e, openai.PermissionDeniedError) and (
            "country" in str(e).lower() or "region" in str(e).lower()
        )

    delay = 3
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - narrow to transient below, else re-raise
            if not (isinstance(e, transient) or _is_geo_403(e)):
                raise
            if attempt == retries - 1:
                raise
            logger.warning("%s failed (%s); retry %d/%d in %ds", what, type(e).__name__,
                           attempt + 1, retries, delay)
            time.sleep(delay)
            delay = min(delay * 2, 60)


def poll(client, job_id, poll_interval):
    while True:
        job = _with_retry(lambda: client.batches.retrieve(job_id), f"retrieve {job_id}")
        counts = getattr(job, "request_counts", None)
        logger.info("Batch %s status=%s counts=%s", job_id, job.status,
                    f"{getattr(counts,'completed',None)}/{getattr(counts,'total',None)} "
                    f"(failed={getattr(counts,'failed',None)})" if counts else "n/a")
        if job.status == _TERMINAL_OK:
            return job
        if job.status in _TERMINAL_BAD:
            raise RuntimeError(f"Batch job {job_id} ended in {job.status}: {getattr(job, 'errors', None)}")
        time.sleep(poll_interval)


def _extract_text(result_line):
    """Pull the assistant text out of one batch output line.

    Output lines look like {custom_id, response: {status_code, body}, error}. A
    non-2xx status or an error object means this request failed. Returns
    (text, failed)."""
    error = result_line.get("error")
    response = result_line.get("response")
    if error or response is None:
        return "", True
    status = response.get("status_code")
    body = response.get("body") or {}
    if status is None or status >= 300:
        return "", True
    try:
        content = body["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return "", True
    return content.strip(), False


def _iter_output_lines(client, job):
    """Yield parsed dicts from the batch output file (and any error-file lines,
    which carry the same {custom_id, error} shape)."""
    for attr in ("output_file_id", "error_file_id"):
        file_id = getattr(job, attr, None)
        if not file_id:
            continue
        text = _with_retry(lambda fid=file_id: client.files.content(fid).text, f"download {file_id}")
        for line in text.splitlines():
            line = line.strip()
            if line:
                yield json.loads(line)


def collect_responses(client, job, responses, failed_flags):
    """Merge one completed job's output lines into the response/flag dicts."""
    for result in _iter_output_lines(client, job):
        cid = result.get("custom_id")
        if cid is None:
            continue
        text, failed = _extract_text(result)
        responses[cid] = text
        failed_flags[cid] = failed


def write_outputs_from_responses(manifest, responses, failed_flags, args, slug):
    model = _strip_provider_prefix(args.model)
    by_id = {rec["custom_id"]: rec for rec in manifest}
    missing = [cid for cid in by_id if cid not in responses]
    if missing:
        logger.warning("%d/%d requests had NO result line across all chunks", len(missing), len(by_id))

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
                cid = rec["custom_id"]
                failed = True if cid in missing else failed_flags.get(cid, True)
                total_failed += int(failed)
                output_example = deepcopy(rec["example"])
                output_example["model_prompt"] = rec["prompt"]
                output_example["model_documents"] = [dataclasses.asdict(d) for d in rec["documents"]]
                output_example["model_answer"] = responses.get(cid, "")
                output_example["model_answer_failed"] = failed
                output_example["model"] = model
                output_example["model_temperature"] = args.temperature
                output_example["model_top_p"] = args.top_p
                output_example["model_prompt_mention_random_ordering"] = args.prompt_mention_random_ordering
                output_example["model_use_random_ordering"] = False
                f.write(json.dumps(output_example) + "\n")
        logger.info("Wrote %d predictions -> %s", len(recs), out_path)

    logger.info("DONE: %d outputs across %d files; %d failed requests",
                len(manifest), len(groups), total_failed)
    return total_failed


def _save_state(state_path, state):
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def resolve_inputs(args):
    inputs = list(args.input_path or [])
    if args.input_glob:
        inputs.extend(sorted(globmod.glob(args.input_glob)))
    seen, ordered = set(), []
    for p in inputs:
        if p not in seen:
            seen.add(p)
            ordered.append(p)
    return ordered


def _state_dict(args, slug, job_ids):
    return {
        "job_ids": job_ids,
        "model": args.model,
        "slug": slug,
        "output_dir": args.output_dir,
        "input_paths": args._input_paths,
        "closedbook": args.closedbook,
        "max_examples": args.max_examples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "prompt_mention_random_ordering": args.prompt_mention_random_ordering,
        "query_aware_contextualization": args.query_aware_contextualization,
    }


def main():
    logging.basicConfig(format="%(asctime)s - %(module)s - %(levelname)s - %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-path", action="append", help="Input file (repeatable).")
    parser.add_argument("--input-glob", help="Glob of input files (e.g. '.../gold_at_*.jsonl.gz').")
    parser.add_argument("--model", default="gpt-4.1")
    parser.add_argument("--output-dir", required=True, help="Directory for per-position prediction files.")
    parser.add_argument("--closedbook", action="store_true")
    parser.add_argument("--prompt-mention-random-ordering", action="store_true")
    parser.add_argument("--query-aware-contextualization", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-enqueued-tokens", type=int, default=1_200_000,
                        help="Per-chunk estimated enqueued-token budget; keep under the org's batch cap "
                             "(gpt-4.1 low tier = 1,350,000). Default 1.2M leaves headroom over the estimate.")
    parser.add_argument("--mode", choices=["run", "fetch"], default="run")
    parser.add_argument("--state-file", default=None, help="Where to persist/resume job state.")
    parser.add_argument("--poll-interval", type=int, default=30)
    args = parser.parse_args()

    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key) if api_key else OpenAI()
    slug = safe_model_slug(args.model)
    model = _strip_provider_prefix(args.model)
    state_path = args.state_file or os.path.join(args.output_dir, f"_batch_state_{slug}.json")

    logger.info("running %s", " ".join(sys.argv))

    if args.mode == "fetch":
        with open(state_path, encoding="utf-8") as f:
            state = json.load(f)
        ns = argparse.Namespace(**{**vars(args), **{
            "output_dir": state["output_dir"], "closedbook": state["closedbook"],
            "max_examples": state["max_examples"], "temperature": state["temperature"],
            "top_p": state["top_p"], "max_new_tokens": state["max_new_tokens"], "model": state["model"],
            "prompt_mention_random_ordering": state["prompt_mention_random_ordering"],
            "query_aware_contextualization": state["query_aware_contextualization"],
        }})
        manifest = build_manifest(state["input_paths"], ns.closedbook, ns.max_examples,
                                  ns.prompt_mention_random_ordering, ns.query_aware_contextualization)
        responses, failed_flags = {}, {}
        for job_id in state["job_ids"]:
            job = poll(client, job_id, args.poll_interval)
            collect_responses(client, job, responses, failed_flags)
        write_outputs_from_responses(manifest, responses, failed_flags, ns, state["slug"])
        return

    input_paths = resolve_inputs(args)
    if not input_paths:
        parser.error("No inputs: pass --input-path and/or --input-glob.")
    args._input_paths = input_paths
    logger.info("Building manifest from %d input file(s)", len(input_paths))
    manifest = build_manifest(input_paths, args.closedbook, args.max_examples,
                              args.prompt_mention_random_ordering, args.query_aware_contextualization)
    chunks = chunk_manifest(manifest, args.max_enqueued_tokens, args.max_new_tokens)
    logger.info("Manifest: %d requests -> %d chunk(s) under %d enqueued tokens each",
                len(manifest), len(chunks), args.max_enqueued_tokens)

    work_dir = pathlib.Path(state_path).parent
    work_dir.mkdir(parents=True, exist_ok=True)

    # Resume: if a state file from a matching run exists, its recorded job_ids map
    # (in order) to the first N chunks — chunking is deterministic. Re-poll those
    # (completed jobs return instantly) and submit only the remaining chunks. This
    # makes a mid-run crash (e.g. a transient timeout) cheap to recover from.
    responses, failed_flags, job_ids = {}, {}, []
    start_chunk = 0
    if os.path.exists(state_path):
        try:
            with open(state_path, encoding="utf-8") as f:
                prev = json.load(f)
            matches = (prev.get("slug") == slug and prev.get("model") == args.model
                       and prev.get("input_paths") == input_paths
                       and prev.get("max_examples") == args.max_examples)
            if matches and prev.get("job_ids"):
                job_ids = list(prev["job_ids"])
                start_chunk = min(len(job_ids), len(chunks))
                logger.info("Resuming: %d chunk(s) already submitted; re-polling them.", len(job_ids))
                for k, job_id in enumerate(job_ids):
                    job = poll(client, job_id, args.poll_interval)
                    collect_responses(client, job, responses, failed_flags)
                    logger.info("Resumed chunk %d/%d (%d responses collected)", k + 1, len(chunks), len(responses))
            else:
                logger.info("State file present but does not match this run; starting fresh.")
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Could not read state file (%s); starting fresh.", e)

    # Submit remaining chunks SEQUENTIALLY: poll each to completion before submitting
    # the next, so only one batch is ever enqueued and the live token total stays
    # under the cap.
    for i in range(start_chunk, len(chunks)):
        job_id, _uploaded = submit_chunk(client, chunks[i], model, args, slug, i, work_dir)
        job_ids.append(job_id)
        _save_state(state_path, _state_dict(args, slug, job_ids))  # persist after each submit, for resume
        job = poll(client, job_id, args.poll_interval)
        collect_responses(client, job, responses, failed_flags)
        logger.info("Chunk %d/%d complete (%d/%d responses collected)",
                    i + 1, len(chunks), len(responses), len(manifest))

    write_outputs_from_responses(manifest, responses, failed_flags, args, slug)
    logger.info("finished running %s", sys.argv[0])


if __name__ == "__main__":
    main()
