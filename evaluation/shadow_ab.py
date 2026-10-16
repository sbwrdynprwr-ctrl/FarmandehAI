"""Paper-only shadow evaluation and paired A/B metrics.

This module deliberately records candidate signals and compares supplied
outcomes; it does not place, route, or simulate orders. It is an isolated
evaluation primitive and is not wired into the trading orchestrator.
"""
from __future__ import annotations

from math import isfinite
from typing import Iterable


VALID_SIGNALS = {"BUY", "SELL", "HOLD"}


def evaluate_shadow_signal(
    *,
    candidate: str,
    signal: str,
    timestamp: str,
    risk_ok: bool,
    spread: float | None = None,
    max_spread: float | None = None,
) -> dict:
    """Return an auditable shadow record; never an executable order."""
    candidate_name = str(candidate).strip()
    normalized_signal = str(signal).strip().upper()
    reasons: list[str] = []
    if not candidate_name:
        reasons.append("missing_candidate")
    if normalized_signal not in VALID_SIGNALS:
        reasons.append("invalid_signal")
    if not timestamp:
        reasons.append("missing_timestamp")
    if not risk_ok:
        reasons.append("risk_check_failed")
    if spread is not None and (not isfinite(spread) or spread < 0):
        reasons.append("invalid_spread")
    if max_spread is not None:
        if not isfinite(max_spread) or max_spread < 0:
            reasons.append("invalid_max_spread")
        elif spread is None:
            reasons.append("spread_unavailable")
        elif spread > max_spread:
            reasons.append("spread_limit_exceeded")

    eligible = not reasons and normalized_signal != "HOLD"
    return {
        "mode": "SHADOW_ONLY",
        "candidate": candidate_name,
        "signal": normalized_signal,
        "timestamp": str(timestamp),
        "eligible_for_shadow_review": eligible,
        "reasons": reasons,
        "order_submitted": False,
        "live": False,
        "real": False,
    }


def _metrics(values: list[float]) -> dict:
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for value in values:
        equity += value
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
    return {
        "samples": len(values),
        "total_r": sum(values),
        "expectancy_r": sum(values) / len(values) if values else 0.0,
        "win_rate": sum(value > 0 for value in values) / len(values) if values else 0.0,
        "max_drawdown_r": max_drawdown,
    }


def compare_ab(
    baseline_returns: Iterable[float],
    candidate_returns: Iterable[float],
    *,
    min_samples: int = 100,
) -> dict:
    """Compare paired, already-observed R outcomes without trading.

    Promotion is intentionally conservative and does not override any existing
    research validation gate. Equal-length paired samples are mandatory.
    """
    baseline = [float(x) for x in baseline_returns]
    candidate = [float(x) for x in candidate_returns]
    if min_samples < 1:
        raise ValueError("min_samples must be at least 1")
    if len(baseline) != len(candidate):
        raise ValueError("A/B inputs must be paired and equal length")
    if any(not isfinite(x) for x in baseline + candidate):
        raise ValueError("A/B outcomes must be finite numbers")

    baseline_metrics = _metrics(baseline)
    candidate_metrics = _metrics(candidate)
    delta = [c - b for b, c in zip(baseline, candidate)]
    delta_mean = sum(delta) / len(delta) if delta else 0.0
    checks = {
        "minimum_sample": len(candidate) >= min_samples,
        "candidate_positive_expectancy": candidate_metrics["expectancy_r"] > 0,
        "candidate_beats_baseline": delta_mean > 0,
    }
    return {
        "mode": "OFFLINE_AB_ANALYSIS",
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "paired_mean_delta_r": delta_mean,
        "checks": checks,
        "promotion_candidate": all(checks.values()),
        "execution_enabled": False,
        "live": False,
        "real": False,
    }
