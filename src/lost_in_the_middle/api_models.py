#!/usr/bin/env python3
"""Unified, provider-agnostic chat-completion layer for the API-based experiments.

The original "Lost in the Middle" code ran open-weight models (MPT, LongChat,
LLaMA-2) locally on A100 GPUs. This module is the modern equivalent: it sends the
exact same prompts (produced by ``lost_in_the_middle.prompting``) to hosted
frontier models through their official SDKs, so the rest of the pipeline
(prompting + evaluation) is reused unchanged.

Supported providers (auto-detected from the model name, or set explicitly):

- ``openai``    e.g. ``gpt-5.1``, ``gpt-4o``                     (env: OPENAI_API_KEY)
- ``anthropic`` e.g. ``claude-opus-4-5``, ``claude-sonnet-4-5`` (env: ANTHROPIC_API_KEY)
- ``google``    e.g. ``gemini-2.5-pro``, ``gemini-2.5-flash``   (env: GEMINI_API_KEY / GOOGLE_API_KEY)

Adding another OpenAI-compatible provider (DeepSeek, Mistral, xAI, Together, a
local vLLM server, ...) is a one-line entry in ``OPENAI_COMPATIBLE_PROVIDERS``.

Each provider client is created lazily and cached, so importing this module has
no side effects and missing SDKs only error if that provider is actually used.
"""
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import List, Optional

from tqdm import tqdm

logger = logging.getLogger(__name__)

# Load API keys from a local .env file if python-dotenv is available. This is a
# no-op when the package isn't installed or no .env exists, and it never
# overrides variables already present in the real environment.
try:
    from dotenv import load_dotenv

    load_dotenv(override=False)
except ImportError:
    pass

# Sentinel returned for a prompt that exhausted all retries, so a single bad
# example never aborts a long (and expensive) run.
FAILED_GENERATION = ""

# Reasoning models (gpt-5.x, o-series) spend tokens on hidden reasoning *before*
# they emit the visible answer, and that reasoning is billed against
# `max_completion_tokens`. If we only granted `max_new_tokens` (~100), the
# reasoning would consume the entire budget, the request would stop with
# finish_reason="length", and `message.content` would come back empty -- making
# a capable model look like it scores ~0. So we add headroom on top of the
# requested answer length for these models. See REASONING_TOKENS.md for the full
# mechanism and why the default below is paired with a low reasoning effort.
DEFAULT_REASONING_TOKEN_BUFFER = 4000
# Backwards-compatible alias.
REASONING_TOKEN_BUFFER = DEFAULT_REASONING_TOKEN_BUFFER

# Default reasoning effort for reasoning models. This benchmark asks for a short
# needle-in-haystack answer, not a hard derivation, so a low effort keeps the
# hidden-token spend (and thus the chance of truncating the visible answer) down
# while leaving accuracy essentially unchanged. Set to None to omit the param.
DEFAULT_REASONING_EFFORT = "low"

# Model-name prefixes that identify OpenAI reasoning models.
_REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def _is_reasoning_model(model: str) -> bool:
    return any(model.startswith(p) for p in _REASONING_PREFIXES)


def _is_gemini_thinking_model(model: str) -> bool:
    """Whether a Gemini model spends hidden "thinking" tokens before its answer.

    The Gemini 2.5 family thinks by default, and those thinking tokens are billed
    against ``max_output_tokens`` -- exactly like OpenAI reasoning models. So a
    naive ``max_output_tokens=100`` is consumed by thinking and the visible answer
    comes back empty (finish_reason=MAX_TOKENS), scoring ~0. We grant these models
    the same answer-budget-plus-reasoning-buffer treatment. Older 1.5/2.0
    (non-thinking) models are excluded so we don't send them a thinking config.
    """
    m = model.lower()
    return m.startswith("gemini-2.5") or "thinking" in m

# Extra OpenAI-compatible providers reachable through the OpenAI SDK by pointing
# at a different base_url. Maps a provider key -> (base_url, api_key_env_vars).
OPENAI_COMPATIBLE_PROVIDERS = {
    "deepseek": ("https://api.deepseek.com", ["DEEPSEEK_API_KEY"]),
    "xai": ("https://api.x.ai/v1", ["XAI_API_KEY"]),
    "mistral": ("https://api.mistral.ai/v1", ["MISTRAL_API_KEY"]),
    "together": ("https://api.together.xyz/v1", ["TOGETHER_API_KEY"]),
}


def detect_provider(model_name: str) -> str:
    """Best-effort provider inference from a model identifier.

    Use ``provider:model`` (e.g. ``deepseek:deepseek-chat``) to be explicit.
    """
    if ":" in model_name:
        provider = model_name.split(":", 1)[0].lower()
        return provider
    lowered = model_name.lower()
    if lowered.startswith("gpt") or lowered.startswith("o1") or lowered.startswith("o3") or lowered.startswith("o4"):
        return "openai"
    if lowered.startswith("claude"):
        return "anthropic"
    if lowered.startswith("gemini"):
        return "google"
    for provider in OPENAI_COMPATIBLE_PROVIDERS:
        if lowered.startswith(provider):
            return provider
    raise ValueError(
        f"Could not infer a provider from model name '{model_name}'. "
        "Prefix it explicitly, e.g. 'openai:gpt-5.1' or 'deepseek:deepseek-chat'."
    )


def _strip_provider_prefix(model_name: str) -> str:
    return model_name.split(":", 1)[1] if ":" in model_name else model_name


def uses_sampling_params(model_name: str, provider: Optional[str] = None) -> bool:
    """Whether ``temperature`` / ``top_p`` are actually sent for this model.

    OpenAI reasoning models (gpt-5.x / o-series) reject sampling params, so we
    omit them; recording them as if they were applied would be misleading. For
    every other model they are sent as given.
    """
    provider = provider or detect_provider(model_name)
    model = _strip_provider_prefix(model_name)
    if provider == "openai" or provider in OPENAI_COMPATIBLE_PROVIDERS:
        return not _is_reasoning_model(model)
    return True


def _http_status(error: Exception) -> Optional[int]:
    """Best-effort extraction of an HTTP status code from a provider SDK error.

    The SDKs disagree on where they put it:
    - OpenAI / Anthropic ``APIStatusError`` -> ``error.status_code`` (instance attr)
    - Google ``google.genai.errors.APIError`` -> ``error.code`` (``status_code``
      does NOT exist there, so the older ``status_code``-only check silently
      treated every Gemini error as transient and burned the full retry budget).

    Returns the status as an int when one can be found, else ``None``.
    """
    for attr in ("status_code", "code"):
        value = getattr(error, attr, None)
        if isinstance(value, int):
            return value
        # Some SDKs stash a numeric string; accept that too.
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return None


def _is_retryable(error: Exception) -> bool:
    """Retry only transient failures.

    A 429 (rate limit) or any 5xx is worth retrying; other 4xx (401 bad key, 404
    unknown model, 400 bad request) will never succeed, so we fail fast instead of
    waiting through the full backoff schedule. Errors without a discoverable status
    code (e.g. network/timeouts) are treated as transient.
    """
    status = _http_status(error)
    if status is not None and 400 <= status < 500 and status != 429:
        return False
    return True


def _require_env(names: List[str]) -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    raise EnvironmentError(
        f"Missing API key. Set one of these environment variables: {', '.join(names)}"
    )


@lru_cache(maxsize=None)
def _openai_client(base_url: Optional[str], api_key_env: tuple):
    from openai import OpenAI

    return OpenAI(api_key=_require_env(list(api_key_env)), base_url=base_url)


@lru_cache(maxsize=None)
def _anthropic_client():
    from anthropic import Anthropic

    return Anthropic(api_key=_require_env(["ANTHROPIC_API_KEY"]))


@lru_cache(maxsize=None)
def _google_client():
    from google import genai

    return genai.Client(api_key=_require_env(["GEMINI_API_KEY", "GOOGLE_API_KEY"]))


def _generate_openai(
    model,
    prompt,
    temperature,
    top_p,
    max_new_tokens,
    base_url,
    api_key_env,
    reasoning_effort=DEFAULT_REASONING_EFFORT,
    reasoning_buffer=DEFAULT_REASONING_TOKEN_BUFFER,
):
    client = _openai_client(base_url, tuple(api_key_env))
    # Newer OpenAI "reasoning" models (gpt-5.x, o-series) reject sampling params
    # like temperature/top_p and use `max_completion_tokens` instead of
    # `max_tokens`. Detect them and adjust the request accordingly.
    is_reasoning = _is_reasoning_model(model)
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if is_reasoning:
        # Grant the visible-answer budget plus headroom for hidden reasoning.
        # `reasoning_effort` caps how much hidden thinking the model does, which
        # is the real driver of whether the answer fits -- see REASONING_TOKENS.md.
        kwargs["max_completion_tokens"] = max_new_tokens + reasoning_buffer
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
    else:
        kwargs["max_tokens"] = max_new_tokens
        kwargs["temperature"] = temperature
        kwargs["top_p"] = top_p
    response = client.chat.completions.create(**kwargs)
    choice = response.choices[0]
    content = (choice.message.content or "").strip()
    # An empty answer that stopped on "length" means the (reasoning) budget ran
    # out before any visible text was produced -- surface it instead of silently
    # scoring it as a wrong answer.
    if not content and getattr(choice, "finish_reason", None) == "length":
        logger.warning(
            "Model '%s' returned EMPTY content with finish_reason='length' -- the "
            "token budget was exhausted (almost certainly by hidden reasoning) "
            "before any visible answer was emitted. This example is effectively "
            "lost and will score as wrong. Re-run with a larger --reasoning-buffer "
            "and/or a lower --reasoning-effort. See REASONING_TOKENS.md.",
            model,
        )
    return content


def _generate_anthropic(model, prompt, temperature, top_p, max_new_tokens):
    client = _anthropic_client()
    # Anthropic models reject specifying BOTH temperature and top_p
    # ("`temperature` and `top_p` cannot both be specified for this model").
    # Send exactly one: prefer an explicit nucleus value (top_p < 1.0), otherwise
    # use temperature. For the benchmark default (temperature=0.0, top_p=1.0) this
    # sends temperature=0.0, i.e. greedy decoding.
    sampling = {"top_p": top_p} if (top_p is not None and top_p < 1.0) else {"temperature": temperature}
    response = client.messages.create(
        model=model,
        max_tokens=max_new_tokens,
        messages=[{"role": "user", "content": prompt}],
        **sampling,
    )
    parts = [block.text for block in response.content if getattr(block, "type", None) == "text"]
    return "".join(parts).strip()


def _generate_google(
    model,
    prompt,
    temperature,
    top_p,
    max_new_tokens,
    reasoning_buffer=DEFAULT_REASONING_TOKEN_BUFFER,
):
    from google.genai import types

    client = _google_client()
    config_kwargs = {
        "temperature": temperature,
        "top_p": top_p,
        "max_output_tokens": max_new_tokens,
    }
    # Gemini 2.5 thinks before answering, and thinking is billed against
    # max_output_tokens. Give it the answer budget plus headroom for thinking, and
    # cap the thinking itself so it can't eat the whole budget and return empty.
    # See REASONING_TOKENS.md (the Gemini analog of the OpenAI reasoning case).
    if _is_gemini_thinking_model(model):
        config_kwargs["max_output_tokens"] = max_new_tokens + reasoning_buffer
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=reasoning_buffer)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(**config_kwargs),
    )
    # `response.text` is a convenience accessor that RAISES when the response has
    # no usable parts (e.g. a safety block, or finish_reason=MAX_TOKENS with no
    # text). Read the candidate parts defensively so a deterministic empty result
    # returns "" instead of throwing and burning the whole retry budget.
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        text = "".join(getattr(part, "text", "") or "" for part in parts).strip()
        if text:
            return text
    if candidates:
        finish_reason = getattr(candidates[0], "finish_reason", None)
        logger.warning(
            "Google model '%s' returned no text (finish_reason=%s); treating as "
            "an empty answer. If finish_reason is MAX_TOKENS this is thinking-token "
            "starvation -- raise --reasoning-buffer. Otherwise check for a safety "
            "block. See REASONING_TOKENS.md.",
            model,
            finish_reason,
        )
    return ""


def _generate_with_status(
    model_name: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    provider: Optional[str],
    max_retries: int,
    reasoning_effort: Optional[str],
    reasoning_buffer: int,
) -> "tuple[str, bool]":
    """Generate one completion and report whether it FAILED (vs. succeeded).

    Returns ``(text, failed)``. ``failed`` is True only when every retry was
    exhausted or a non-retryable error was hit -- it is *not* set for a model
    that legitimately returns an empty string. This lets callers flag genuine
    failures in their output without conflating them with empty answers.
    """
    provider = provider or detect_provider(model_name)
    model = _strip_provider_prefix(model_name)

    def _dispatch():
        if provider == "openai":
            return _generate_openai(
                model, prompt, temperature, top_p, max_new_tokens, None, ["OPENAI_API_KEY"],
                reasoning_effort=reasoning_effort, reasoning_buffer=reasoning_buffer,
            )
        if provider == "anthropic":
            return _generate_anthropic(model, prompt, temperature, top_p, max_new_tokens)
        if provider == "google":
            return _generate_google(
                model, prompt, temperature, top_p, max_new_tokens, reasoning_buffer=reasoning_buffer,
            )
        if provider in OPENAI_COMPATIBLE_PROVIDERS:
            base_url, api_key_env = OPENAI_COMPATIBLE_PROVIDERS[provider]
            return _generate_openai(
                model, prompt, temperature, top_p, max_new_tokens, base_url, api_key_env,
                reasoning_effort=reasoning_effort, reasoning_buffer=reasoning_buffer,
            )
        raise ValueError(f"Unknown provider: {provider}")

    last_error = None
    for attempt in range(max_retries):
        try:
            return _dispatch(), False
        except Exception as error:  # noqa: BLE001 - we want to retry on any transient API error
            last_error = error
            if not _is_retryable(error):
                logger.error(
                    "Non-retryable error for model '%s' (%s); giving up on this prompt. "
                    "Check the model name and API key.",
                    model_name,
                    error,
                )
                return FAILED_GENERATION, True
            sleep_seconds = min(2**attempt, 30)
            logger.warning(
                "Generation attempt %d/%d failed (%s); retrying in %ds",
                attempt + 1,
                max_retries,
                error,
                sleep_seconds,
            )
            time.sleep(sleep_seconds)
    logger.error("Giving up on a prompt after %d attempts. Last error: %s", max_retries, last_error)
    return FAILED_GENERATION, True


def generate_single(
    model_name: str,
    prompt: str,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_new_tokens: int = 100,
    provider: Optional[str] = None,
    max_retries: int = 5,
    reasoning_effort: Optional[str] = DEFAULT_REASONING_EFFORT,
    reasoning_buffer: int = DEFAULT_REASONING_TOKEN_BUFFER,
) -> str:
    """Generate one completion, retrying transient errors with exponential backoff."""
    text, _failed = _generate_with_status(
        model_name, prompt, temperature, top_p, max_new_tokens, provider, max_retries,
        reasoning_effort, reasoning_buffer,
    )
    return text


def generate_responses(
    model_name: str,
    prompts: List[str],
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_new_tokens: int = 100,
    num_workers: int = 4,
    provider: Optional[str] = None,
    max_retries: int = 5,
    reasoning_effort: Optional[str] = DEFAULT_REASONING_EFFORT,
    reasoning_buffer: int = DEFAULT_REASONING_TOKEN_BUFFER,
    return_failures: bool = False,
):
    """Generate completions for a list of prompts, optionally in parallel.

    Order is preserved. ``num_workers`` parallel requests trade speed against
    provider rate limits; lower it if you see frequent 429s.

    Returns a ``List[str]`` of responses by default. With ``return_failures=True``
    returns ``(responses, failures)`` where ``failures`` is a parallel
    ``List[bool]`` marking prompts whose generation failed (so callers can record
    an explicit per-row failure flag rather than guessing from an empty string).
    """
    provider = provider or detect_provider(model_name)
    logger.info("Using provider '%s' for model '%s'", provider, model_name)

    def _run(prompt):
        return _generate_with_status(
            model_name, prompt, temperature, top_p, max_new_tokens, provider, max_retries,
            reasoning_effort, reasoning_buffer,
        )

    if num_workers <= 1:
        results = [_run(prompt) for prompt in tqdm(prompts)]
    else:
        results: List[Optional["tuple[str, bool]"]] = [None] * len(prompts)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_run, prompt): idx for idx, prompt in enumerate(prompts)}
            for future in tqdm(as_completed(futures), total=len(futures)):
                results[futures[future]] = future.result()

    responses = [text for text, _failed in results]
    failures = [failed for _text, failed in results]

    num_failed = sum(failures)
    if num_failed:
        logger.warning(
            "%d/%d prompts produced a FAILED generation (retries exhausted / "
            "non-retryable error). These count as wrong answers in scoring -- "
            "inspect the logs above before trusting the metrics.",
            num_failed,
            len(responses),
        )
    if return_failures:
        return responses, failures
    return responses
