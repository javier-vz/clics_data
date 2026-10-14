"""Exact conditional uniform-surjection null for observed A associations and V active forms.

The assignments are assumed labeled and independently exchangeable before conditioning.
Conditioning on *all V bins occupied* yields P(rho) ∝ 1/prod_i (rho_i!),
subject to rho_i >= 1 and sum_i rho_i = A.

An EXACT independent sampler uses iid zero-truncated Poisson occupancies
conditioned on their sum. The Poisson rate cancels after conditioning;
we tune it only to maximize acceptance of the rejection sampler.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq


def concentration(counts: np.ndarray) -> float:
    a = int(np.sum(counts))
    if a < 2:
        return 0.0
    return float(np.dot(counts.astype(np.float64), counts.astype(np.float64)) - a) / (a * (a - 1))


def minimal_concentration(a: int, v: int) -> float:
    if a < v or v < 1:
        raise ValueError("Need a >= v >= 1")
    if a < 2:
        return 0.0
    q, r = divmod(a, v)
    excess_pairs = (v - r) * q * (q - 1) + r * (q + 1) * q
    return excess_pairs / (a * (a - 1))


def zt_poisson_rate(a: int, v: int) -> float:
    if a == v:
        return 0.0
    mean = a / v
    def residual(lam: float) -> float:
        return lam / (-math.expm1(-lam)) - mean
    return brentq(residual, 1e-12, max(5.0, mean + 5), xtol=1e-12)


def _excess_cdf(lam: float, d: int) -> np.ndarray:
    # Zero-truncated Poisson: P(rho=1+k) = lam^(k+1)/((k+1)!*(exp(lam)-1)).
    # Occupancies rho > d+1 cannot appear when total excess is exactly d;
    # omitting them does not change the desired CONDITIONAL law.
    p = [lam / math.expm1(lam)]
    for k in range(1, d+1):
        p.append(p[-1] * lam / (k + 1))
        if p[-1] == 0.0:
            break
    probs = np.asarray(p, dtype=float)
    probs /= probs.sum()
    cdf = np.cumsum(probs)
    cdf[-1] = 1.0
    return cdf


def sample_surjection_counts(
    a: int,
    v: int,
    samples: int,
    seed: int,
    batch_size: int = 64,
    max_proposals: int = 2_000_000,
) -> np.ndarray:
    """IID exact samples of counts from uniform surjections A -> V, A>=V.

    Unlike generic random occupancy, all V observed forms are occupied in every draw.
    Raises an explicit error if its exact rejection budget is insufficient.
    """
    if not (a >= v >= 1 and samples >= 1):
        raise ValueError("Need a>=v>=1 and samples>=1")
    if a == v:
        return np.ones((samples, v), dtype=np.int32)
    d = a - v
    rng = np.random.default_rng(seed)
    cdf = _excess_cdf(zt_poisson_rate(a, v), d)
    draws = np.empty((samples, v), dtype=np.int32)
    collected = 0
    attempted = 0
    while collected < samples:
        b = min(batch_size, max_proposals - attempted)
        if b <= 0:
            raise RuntimeError(
                f"Conditional-surjection sampler exhausted {max_proposals:,} proposals for A={a}, V={v}. "
                "Increase max_proposals or use fewer samples."
            )
        extras = np.searchsorted(cdf, rng.random((b, v)), side="left")
        accepted = extras[np.sum(extras, axis=1) == d]
        n = min(len(accepted), samples - collected)
        if n:
            draws[collected:collected+n] = accepted[:n].astype(np.int32) + 1
            collected += n
        attempted += b
    assert np.all(draws >= 1) and np.all(draws.sum(axis=1) == a)
    return draws


def occupancy_draw_metrics(draws: np.ndarray) -> dict[str, np.ndarray]:
    a = int(draws[0].sum())
    sum_sq = np.einsum("ij,ij->i", draws.astype(np.int64), draws.astype(np.int64))
    h = (sum_sq - a) / (a * (a - 1)) if a > 1 else np.zeros(len(draws))
    return {
        "h_null": np.asarray(h, dtype=float),
        "max_rho_null": draws.max(axis=1).astype(int),
        "forms_ge3_null": (draws >= 3).sum(axis=1).astype(int),
        "forms_ge2_null": (draws >= 2).sum(axis=1).astype(int),
    }


def monte_carlo_tail(value: float, null: np.ndarray, side: str) -> float:
    """Conservative finite-replicate tail p-value with the +1 correction."""
    if side == "lower":
        extreme = np.count_nonzero(null <= value)
    elif side == "upper":
        extreme = np.count_nonzero(null >= value)
    else:
        raise ValueError("side must be lower or upper")
    return (extreme + 1) / (len(null) + 1)
