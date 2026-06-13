# Running modern API models (OpenAI / Anthropic / Google / OpenAI-compatible)

> **New in this fork.** The original experiments below run open-weight models
> (MPT, LongChat, LLaMA-2) locally on A100-class GPUs. This section adds an
> API-based path that sends the **exact same prompts** to hosted frontier models
> and reuses the **same evaluation scripts**, so no GPU is required. This is in
> the spirit of the original paper, which also reported GPT-3.5 and Claude
> results via API.

### Setup

Install the lightweight requirements (no CUDA needed). The base install already
includes the provider SDKs (`openai`, `anthropic`, `google-genai`) and
`python-dotenv`, so this is all you need:

```
pip install -e .
```

Set the API key for whichever provider(s) you want to use. The easiest way is to
copy `.env.example` to `.env` and fill in your key(s) — it is gitignored and
loaded automatically:

```
cp .env.example .env
# then edit .env and set, e.g., OPENAI_API_KEY=sk-...
```

Alternatively, export them in your shell:

```
export OPENAI_API_KEY=sk-...        # for gpt-5.1, gpt-4o, o3, ...   (Windows PowerShell: $env:OPENAI_API_KEY="sk-...")
export ANTHROPIC_API_KEY=sk-ant-... # for claude-opus-4-5, claude-sonnet-4-5, ...
export GEMINI_API_KEY=...           # for gemini-2.5-pro, gemini-2.5-flash, ...
```

### One-command sweep (recommended)

`scripts/run_api_sweep.py` runs every gold position for both tasks and prints a
summary table — cross-platform, no shell loops needed:

```
python ./scripts/run_api_sweep.py --model gpt-5.1 --max-examples 100
python ./scripts/run_api_sweep.py --model claude-opus-4-5 --task qa --max-examples 50
```

The per-script commands below give you finer control if you prefer.

> **Windows / PowerShell users:** the per-script sections below use bash
> `for … do … done` loops, which do **not** run in PowerShell. Either use
> `run_api_sweep.py` above (fully cross-platform), or translate the loop, e.g.:
>
> ```powershell
> $MODEL = "gpt-5.1"
> foreach ($gold in 0,4,9,14,19) {
>     # NOTE: if $MODEL contains a provider prefix like "deepseek:deepseek-chat",
>     # replace ':' and '/' in the output filename — they are illegal on Windows.
>     $slug = $MODEL -replace '[:/]', '_'
>     python -u ./scripts/get_qa_responses_from_api.py `
>         --input-path "qa_data/20_total_documents/nq-open-20_total_documents_gold_at_$gold.jsonl.gz" `
>         --model $MODEL --max-examples 100 --num-workers 4 `
>         --output-path "qa_predictions/20_total_documents/nq-open-20_total_documents_gold_at_$gold-$slug-predictions.jsonl.gz"
> }
> ```
>
> The bash examples also embed `${MODEL}` directly in output paths; with an
> explicit provider prefix (`deepseek:deepseek-chat`) the `:` is an illegal
> filename character on Windows, so sanitize it as shown (the sweep script does
> this for you).

#### Reasoning models (gpt-5.x / o-series)

These models spend hidden tokens on reasoning *before* the visible answer. The
API scripts handle this automatically (a low `--reasoning-effort` plus a
`--reasoning-buffer` of extra completion tokens). If you see
`finish_reason=length` warnings or many `model_answer_failed` rows, raise
`--reasoning-buffer`. See [REASONING_TOKENS.md](./REASONING_TOKENS.md) for the
full mechanism.

Model names are mapped to a provider automatically (`gpt*`/`o*` → OpenAI,
`claude*` → Anthropic, `gemini*` → Google). To use any other OpenAI-compatible
endpoint (DeepSeek, xAI/Grok, Mistral, Together, a local vLLM server, ...),
prefix the model with the provider, e.g. `--model deepseek:deepseek-chat`
(set `DEEPSEEK_API_KEY`).

### Multi-document QA with an API model

`--max-examples N` runs only the first `N` questions of a file — use it to get a
cheap estimate of the lost-in-the-middle curve before committing to a full run.
`--num-workers` controls how many requests run concurrently (lower it on rate
limits).

```
MODEL=gpt-5.1   # or: claude-opus-4-5, gemini-2.5-pro, deepseek:deepseek-chat, ...
for gold_index in 0 4 9 14 19; do
    python -u ./scripts/get_qa_responses_from_api.py \
        --input-path qa_data/20_total_documents/nq-open-20_total_documents_gold_at_${gold_index}.jsonl.gz \
        --model ${MODEL} \
        --max-examples 100 \
        --num-workers 4 \
        --output-path qa_predictions/20_total_documents/nq-open-20_total_documents_gold_at_${gold_index}-${MODEL}-predictions.jsonl.gz
done

for gold_index in 0 4 9 14 19; do
    python -u ./scripts/evaluate_qa_responses.py \
        --input-path qa_predictions/20_total_documents/nq-open-20_total_documents_gold_at_${gold_index}-${MODEL}-predictions.jsonl.gz \
        --output-path qa_predictions/20_total_documents/nq-open-20_total_documents_gold_at_${gold_index}-${MODEL}-predictions-scored.jsonl.gz
done
```

The reported `best_subspan_em` per gold index traces the characteristic U-shaped
"lost in the middle" curve (highest when the gold document is first or last).

### Key-value retrieval with an API model

```
MODEL=gpt-5.1
for gold_index in 0 34 69 104 139; do
    python -u ./scripts/get_kv_responses_from_api.py \
        --input-path kv_retrieval_data/kv-retrieval-140_keys.jsonl.gz \
        --gold-index ${gold_index} \
        --model ${MODEL} \
        --max-examples 100 \
        --num-workers 4 \
        --output-path kv_predictions/kv-retrieval-140_keys_gold_at_${gold_index}-${MODEL}-predictions.jsonl.gz
done

for gold_index in 0 34 69 104 139; do
    python -u ./scripts/evaluate_kv_responses.py \
        --input-path kv_predictions/kv-retrieval-140_keys_gold_at_${gold_index}-${MODEL}-predictions.jsonl.gz \
        --output-path kv_predictions/kv-retrieval-140_keys_gold_at_${gold_index}-${MODEL}-predictions-scored.jsonl.gz
done
```

---

# Multi-Document Question Answering

Note: all of these experiments were run on one or more A100 GPUs with 80GB of
VRAM. You may need to modify commands to fit your own computing environment
(e.g., changing the batch size, the max memory per GPU, the number of GPUs, etc)

## mpt-30b-instruct

To run `mpt-30b` and `mpt-30b-instruct` on multi-document question answering,
use [`./scripts/get_qa_responses_from_mpt.py`](./scripts/get_qa_responses_from_mpt.py).
Below are commands for running `mpt-30b-instruct` on different multi-document QA
settings.

### mpt-30b-instruct on oracle

Getting predictions:

```
python -u ./scripts/get_qa_responses_from_mpt.py \
    --input-path qa_data/nq-open-oracle.jsonl.gz \
    --num-gpus 1 \
    --max-new-tokens 100 \
    --batch-size 1 \
    --max-memory-per-gpu 80 \
    --num-gpus 1 \
    --model mosaicml/mpt-30b-instruct \
    --output-path qa_predictions/nq-open-oracle-mpt-30b-instruct-predictions.jsonl.gz
```

Evaluating: 

```
python -u ./scripts/evaluate_qa_responses.py \
    --input-path qa_predictions/nq-open-oracle-mpt-30b-instruct-predictions.jsonl.gz \
    --output-path qa_predictions/nq-open-oracle-mpt-30b-instruct-predictions-scored.jsonl.gz
```

You should get something approximately around:

```
best_subspan_em: 0.816572504708098
```

### mpt-30b-instruct on closedbook

Getting predictions:

```
python -u ./scripts/get_qa_responses_from_mpt.py \
    --input-path qa_data/nq-open-oracle.jsonl.gz \
    --num-gpus 1 \
    --max-new-tokens 100 \
    --batch-size 1 \
    --max-memory-per-gpu 80 \
    --num-gpus 1 \
    --closedbook \
    --model mosaicml/mpt-30b-instruct \
    --output-path qa_predictions/nq-open-oracle-mpt-30b-instruct-closedbook-predictions.jsonl.gz
```

Evaluating: 

```
python -u ./scripts/evaluate_qa_responses.py \
    --input-path qa_predictions/nq-open-oracle-mpt-30b-instruct-closedbook-predictions.jsonl.gz \
    --output-path qa_predictions/nq-open-oracle-mpt-30b-instruct-closedbook-predictions-scored.jsonl.gz
```

You should get something approximately around:

```
best_subspan_em: 0.3167608286252354
```

### mpt-30b-instruct on 20-document setting

Getting predictions:

```
for gold_index in 0 4 9 14 19; do
    python -u ./scripts/get_qa_responses_from_mpt.py \
        --input-path qa_data/20_total_documents/nq-open-20_total_documents_gold_at_${gold_index}.jsonl.gz \
        --num-gpus 1 \
        --max-new-tokens 100 \
        --batch-size 1 \
        --max-memory-per-gpu 80 \
        --num-gpus 1 \
        --model mosaicml/mpt-30b-instruct \
        --output-path qa_predictions/20_total_documents/nq-open-20_total_documents_gold_at_${gold_index}-mpt-30b-instruct-predictions.jsonl.gz
done
```

Evaluating: 

```
for gold_index in 0 4 9 14 19; do
    python -u ./scripts/evaluate_qa_responses.py \
        --input-path qa_predictions/20_total_documents/nq-open-20_total_documents_gold_at_${gold_index}-mpt-30b-instruct-predictions.jsonl.gz \
        --output-path qa_predictions/20_total_documents/nq-open-20_total_documents_gold_at_${gold_index}-mpt-30b-instruct-predictions-scored.jsonl.gz
done
```

You should get something approximately around:

```
Gold Index 0:
best_subspan_em: 0.536723163841808

Gold Index 4:
best_subspan_em: 0.5175141242937853

Gold Index 9:
best_subspan_em: 0.5216572504708098

Gold Index 14:
best_subspan_em: 0.5265536723163842

Gold Index 19:
best_subspan_em: 0.5623352165725047
```

## longchat-13b-16k

To run `longchat-13b-16k` on multi-document question answering,
use [`./scripts/get_qa_responses_from_longchat.py`](./scripts/get_qa_responses_from_longchat.py).
Below are commands for running `longchat-13b-16k` on different multi-document QA
settings.

### longchat-13b-16k on oracle

Getting predictions:

```
python -u ./scripts/get_qa_responses_from_longchat.py \
    --input-path qa_data/nq-open-oracle.jsonl.gz \
    --num-gpus 1 \
    --max-new-tokens 100 \
    --batch-size 8 \
    --max-memory-per-gpu 80 \
    --num-gpus 1 \
    --model lmsys/longchat-13b-16k \
    --output-path qa_predictions/nq-open-oracle-longchat-13b-16k-predictions.jsonl.gz
```

Evaluating: 

```
python -u ./scripts/evaluate_qa_responses.py \
    --input-path qa_predictions/nq-open-oracle-longchat-13b-16k-predictions.jsonl.gz \
    --output-path qa_predictions/nq-open-oracle-longchat-13b-16k-predictions-scored.jsonl.gz
```

You should get something approximately around:

```
best_subspan_em: 0.8263653483992467
```

### longchat-13b-16k on closedbook

Getting predictions:

```
python -u ./scripts/get_qa_responses_from_longchat.py \
    --input-path qa_data/nq-open-oracle.jsonl.gz \
    --num-gpus 1 \
    --max-new-tokens 100 \
    --batch-size 8 \
    --max-memory-per-gpu 80 \
    --num-gpus 1 \
    --closedbook \
    --model lmsys/longchat-13b-16k \
    --output-path qa_predictions/nq-open-oracle-longchat-13b-16k-closedbook-predictions.jsonl.gz
```

Evaluating: 

```
python -u ./scripts/evaluate_qa_responses.py \
    --input-path qa_predictions/nq-open-oracle-longchat-13b-16k-closedbook-predictions.jsonl.gz \
    --output-path qa_predictions/nq-open-oracle-longchat-13b-16k-closedbook-predictions-scored.jsonl.gz
```

You should get something approximately around:

```
best_subspan_em: 0.34990583804143127
```

### longchat-13b-16k on 20-document setting

Getting predictions:

```
for gold_index in 0 4 9 14 19; do
    python -u ./scripts/get_qa_responses_from_longchat.py \
        --input-path qa_data/20_total_documents/nq-open-20_total_documents_gold_at_${gold_index}.jsonl.gz \
        --num-gpus 1 \
        --max-new-tokens 100 \
        --batch-size 1 \
        --max-memory-per-gpu 80 \
        --num-gpus 1 \
        --model lmsys/longchat-13b-16k \
        --output-path qa_predictions/20_total_documents/nq-open-20_total_documents_gold_at_${gold_index}-longchat-13b-16k-predictions.jsonl.gz
done
```

Evaluating: 

```
for gold_index in 0 4 9 14 19; do
    python -u ./scripts/evaluate_qa_responses.py \
        --input-path qa_predictions/20_total_documents/nq-open-20_total_documents_gold_at_${gold_index}-longchat-13b-16k-predictions.jsonl.gz \
        --output-path qa_predictions/20_total_documents/nq-open-20_total_documents_gold_at_${gold_index}-longchat-13b-16k-predictions-scored.jsonl.gz
done
```

You should get something approximately around:

```
Gold Index 0:
best_subspan_em: 0.6858757062146893

Gold Index 4:
best_subspan_em: 0.5740112994350283

Gold Index 9:
best_subspan_em: 0.5532956685499059

Gold Index 14:
best_subspan_em: 0.5250470809792843

Gold Index 19:
best_subspan_em: 0.5502824858757062
```

## llama-2

To run llama-2 models on multi-document question answering,
use [`./scripts/get_qa_responses_from_llama_2.py`](./scripts/get_qa_responses_from_llama_2.py).
Below are commands for running `Llama-2-70b-chat-hf` on different multi-document QA
settings. You can run any other Llama-2 model by changing the model identifier (e.g., 
`Llama-2-13b-hf`, `Llama-2-7b-chat-hf`, etc).

Running the 70b models requires 2 80GB GPUs. If you're running a 13b or 7b model, only
1 80GB GPU is required.

### Llama-2-70b-chat-hf on oracle

Getting predictions:

```
python -u ./scripts/get_qa_responses_from_llama_2.py \
    --input-path qa_data/nq-open-oracle.jsonl.gz \
    --max-new-tokens 100 \
    --num-gpus 2 \
    --model meta-llama/Llama-2-70b-chat-hf \
    --output-path qa_predictions/nq-open-oracle-llama-2-70b-chat-hf-predictions.jsonl.gz
```

Evaluating: 

```
python -u ./scripts/evaluate_qa_responses.py \
    --input-path qa_predictions/nq-open-oracle-llama-2-70b-chat-hf-predictions.jsonl.gz \
    --output-path qa_predictions/nq-open-oracle-llama-2-70b-chat-hf-predictions-scored.jsonl.gz
```

You should get something approximately around:

```
best_subspan_em: 0.8467043314500942
```

### Llama-2-70b-chat-hf on closedbook

Getting predictions:

```
python -u ./scripts/get_qa_responses_from_llama_2.py \
    --input-path qa_data/nq-open-oracle.jsonl.gz \
    --num-gpus 2 \
    --max-new-tokens 100 \
    --closedbook \
    --model meta-llama/Llama-2-70b-chat-hf \
    --output-path qa_predictions/nq-open-oracle-llama-2-70b-chat-hf-closedbook-predictions.jsonl.gz
```

Evaluating: 

```
python -u ./scripts/evaluate_qa_responses.py \
    --input-path qa_predictions/nq-open-oracle-llama-2-70b-chat-hf-closedbook-predictions.jsonl.gz \
    --output-path qa_predictions/nq-open-oracle-llama-2-70b-chat-hf-closedbook-predictions-scored.jsonl.gz
```

You should get something approximately around:

```
best_subspan_em: 0.35291902071563086
```

### Llama-2-70b-chat-hf on 20-document setting

Getting predictions:

```
for gold_index in 0 4 9 14 19; do
    python -u ./scripts/get_qa_responses_from_llama_2.py \
        --input-path qa_data/20_total_documents/nq-open-20_total_documents_gold_at_${gold_index}.jsonl.gz \
        --max-new-tokens 100 \
        --num-gpus 2 \
        --model meta-llama/Llama-2-70b-chat-hf \
        --output-path qa_predictions/20_total_documents/nq-open-20_total_documents_gold_at_${gold_index}-llama-2-70b-chat-hf-predictions.jsonl.gz
done
```

Evaluating: 

```
for gold_index in 0 4 9 14 19; do
    python -u ./scripts/evaluate_qa_responses.py \
        --input-path qa_predictions/20_total_documents/nq-open-20_total_documents_gold_at_${gold_index}-llama-2-70b-chat-hf-predictions.jsonl.gz \
        --output-path qa_predictions/20_total_documents/nq-open-20_total_documents_gold_at_${gold_index}-llama-2-70b-chat-hf-predictions-scored.jsonl.gz
done
```

You should get something approximately around:

```
Gold Index 0:
best_subspan_em: 0.567741935483871

Gold Index 4:
best_subspan_em: 0.5332068311195446

Gold Index 9:
best_subspan_em: 0.540796963946869

Gold Index 14:
best_subspan_em: 0.596584440227704

Gold Index 19:
best_subspan_em: 0.6948766603415559
```

# Key-Value Retrieval

Note: all of these experiments were run on one or more A100 GPUs with 80GB of
VRAM. You may need to modify commands to fit your own computing environment
(e.g., changing the batch size, the max memory per GPU, the number of GPUs, etc)

## mpt-30b-instruct

To run `mpt-30b` and `mpt-30b-instruct` on key-value retrieval, use
[`./scripts/get_kv_responses_from_mpt.py`](./scripts/get_kv_responses_from_mpt.py).
Below are commands for running `mpt-30b-instruct` on different KV retrieval
settings.

### mpt-30b-instruct with 140 total key-value pairs

Getting predictions:

```
for gold_index in 0 34 69 104 139; do
    python -u ./scripts/get_kv_responses_from_mpt.py \
        --input-path kv_retrieval_data/kv-retrieval-140_keys.jsonl.gz \
        --batch-size 1 \
        --gold-index ${gold_index} \
        --model mosaicml/mpt-30b-instruct \
        --max-memory-per-gpu 80 \
        --num-gpus 1 \
        --output-path kv_predictions/kv-retrieval-140_keys_gold_at_${gold_index}-mpt-30b-instruct-predictions.jsonl.gz
done
```

Evaluating: 

```
for gold_index in 0 34 69 104 139; do
    python -u ./scripts/evaluate_kv_responses.py \
        --input-path kv_predictions/kv-retrieval-140_keys_gold_at_${gold_index}-mpt-30b-instruct-predictions.jsonl.gz \
        --output-path kv_predictions/kv-retrieval-140_keys_gold_at_${gold_index}-mpt-30b-instruct-predictions-scored.jsonl.gz
done
```

You should get something approximately around:

```
Gold Index 0:
best_subspan_em: 1.0

Gold Index 34:
best_subspan_em: 0.936

Gold Index 69:
best_subspan_em: 0.886

Gold Index 104:
best_subspan_em: 0.804

Gold Index 139:
best_subspan_em: 0.962
```

## longchat-13b-16k

To run `longchat-13b-16k` on key-value retrieval, use
[`./scripts/get_kv_responses_from_longchat.py`](./scripts/get_kv_responses_from_mpt.py).
Below are commands for running `longchat-13b-16k` on different KV retrieval
settings.

### longchat-13b-16k with 140 total key-value pairs

Getting predictions:

```
for gold_index in 0 34 69 104 139; do
    python -u ./scripts/get_kv_responses_from_longchat.py \
        --input-path kv_retrieval_data/kv-retrieval-140_keys.jsonl.gz \
        --batch-size 1 \
        --gold-index ${gold_index} \
        --max-memory-per-gpu 80 \
        --num-gpus 2 \
        --model lmsys/longchat-13b-16k \
        --output-path kv_predictions/kv-retrieval-140_keys_gold_at_${gold_index}-longchat-13b-16k-predictions.jsonl.gz
done
```

Evaluating: 

```
for gold_index in 0 34 69 104 139; do
    python -u ./scripts/evaluate_kv_responses.py \
        --input-path kv_predictions/kv-retrieval-140_keys_gold_at_${gold_index}-longchat-13b-16k-predictions.jsonl.gz \
        --output-path kv_predictions/kv-retrieval-140_keys_gold_at_${gold_index}-longchat-13b-16k-predictions-scored.jsonl.gz
done
```

You should get something approximately around:

```
Gold Index 0:
best_subspan_em: 0.37

Gold Index 34:
best_subspan_em: 0.382

Gold Index 69:
best_subspan_em: 0.354

Gold Index 104:
best_subspan_em: 0.656

Gold Index 139:
best_subspan_em: 0.886
```
