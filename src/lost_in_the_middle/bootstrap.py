"""Bootstrap confidence intervals for the position-curve experiments.

Each (model, doc-count, gold-position) result is the mean of `n` binary
per-example scores (gold span present in the prediction -> 1, else 0) — a
proportion, and at n=100 a noisy one (SE ~= sqrt(p(1-p)/n) ~= 0.043 at p=0.75, so
a 95% CI of ~+/-0.085). The project's headline shapes rest on between-position
spreads of only ~0.07-0.14, i.e. the effect is the same size as the error bar, so
every shape claim is gated on interval estimation. This module provides it.

Two things are computed (see CI_IMPLEMENTATION_HANDOFF.md):

1. **Per-position 95% CI** — percentile bootstrap of each position's mean.
2. **Paired difference vs a reference position** — because the *same questions*
   appear at every gold position (only the gold doc's index moves), we resample
   question indices **once per replicate** and apply that same index set to every
   position before differencing. This paired bootstrap is tighter and more correct
   than differencing two independent CIs, and it is the real test of the shape
   claims: if the difference CI excludes 0 the positions genuinely differ; if it
   straddles 0 that position effect is a non-result.

A `spread` CI (observed-best minus observed-worst position, paired) is also
returned as a single "is there *any* position effect" summary. The best/worst
positions are chosen from the observed means (mildly post-hoc — report as
descriptive), then their difference is bootstrapped at fixed indices to avoid the
per-replicate argmax/argmin selection bias.

The RNG is seeded so results are deterministic across reruns.
"""
import numpy as np

DEFAULT_B = 10000
DEFAULT_SEED = 0


def _percentile_ci(replicates, alpha=0.05):
    lo = float(np.percentile(replicates, 100 * (alpha / 2)))
    hi = float(np.percentile(replicates, 100 * (1 - alpha / 2)))
    return lo, hi


def position_bootstrap(scores_by_position, ordered_positions, ref_position,
                       B=DEFAULT_B, seed=DEFAULT_SEED, alpha=0.05):
    """Bootstrap per-position CIs and paired differences vs a reference position.

    Args:
        scores_by_position: dict ``position -> 1D np.ndarray`` of per-example 0/1
            scores. Every array MUST be aligned to the same questions in the same
            order and have the same length ``n`` (paired bootstrap requirement).
        ordered_positions: positions in plotting order (e.g. sorted gold indices).
        ref_position: the position every paired difference is measured against.
        B: number of bootstrap replicates.
        seed: RNG seed (deterministic output).
        alpha: 1 - confidence (0.05 -> 95% CI).

    Returns:
        (per_position, spread) where
        ``per_position[pos] = {mean, ci_low, ci_high, ci_halfwidth,
            [diff_vs_ref, diff_ci_low, diff_ci_high, diff_excludes_zero]}``
        (the diff_* keys are absent for ``ref_position`` itself) and
        ``spread = {best_position, worst_position, spread, ci_low, ci_high,
            excludes_zero}``.
    """
    positions = list(ordered_positions)
    n = len(scores_by_position[positions[0]])
    if any(len(scores_by_position[p]) != n for p in positions):
        raise ValueError("All positions must have the same number of aligned examples for a paired bootstrap.")

    rng = np.random.default_rng(seed)
    # One shared set of resampled indices per replicate -> paired across positions.
    idx = rng.integers(0, n, size=(B, n))
    reps = {p: np.asarray(scores_by_position[p], dtype=float)[idx].mean(axis=1) for p in positions}
    means = {p: float(np.asarray(scores_by_position[p], dtype=float).mean()) for p in positions}

    per_position = {}
    for p in positions:
        lo, hi = _percentile_ci(reps[p], alpha)
        entry = {"mean": means[p], "ci_low": lo, "ci_high": hi, "ci_halfwidth": (hi - lo) / 2.0}
        if p != ref_position:
            diff = reps[p] - reps[ref_position]
            dlo, dhi = _percentile_ci(diff, alpha)
            entry.update({
                "diff_vs_ref": means[p] - means[ref_position],
                "diff_ci_low": dlo,
                "diff_ci_high": dhi,
                "diff_excludes_zero": bool(dlo > 0.0 or dhi < 0.0),
            })
        per_position[p] = entry

    best = max(means, key=means.get)
    worst = min(means, key=means.get)
    sdiff = reps[best] - reps[worst]
    slo, shi = _percentile_ci(sdiff, alpha)
    spread = {
        "best_position": int(best),
        "worst_position": int(worst),
        "spread": means[best] - means[worst],
        "ci_low": slo,
        "ci_high": shi,
        "excludes_zero": bool(slo > 0.0 or shi < 0.0),
    }
    return per_position, spread
