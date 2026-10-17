"""Paper-only shadow evaluation and paired A/B metrics.

This module deliberately records candidate signals and compares supplied
outcomes; it does not place, route, or simulate orders. It is an isolated
evaluation primitive and is not wired into the trading orchestrator.
"""
from __future__ import annotations

from math import isfinite
from typing import Iterable

import pandas as pd


VALID_SIGNALS = {"BUY", "SELL", "HOLD"}


def _timestamp_key(value: object) -> str:
    """Canonicalize timestamp representations before pairing A/B outcomes.

    Pandas Timestamp, numpy datetime64, and ISO strings can stringify differently
    even when they identify the same UTC bar. Normalize them to a UTC ISO key.
    Naive timestamps are treated as UTC because research data is validated UTC.
    """
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("invalid evaluation timestamp") from exc
    if pd.isna(timestamp):
        raise ValueError("invalid evaluation timestamp")
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.isoformat()


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
    candidate_name = "" if candidate is None else str(candidate).strip()
    normalized_signal = str(signal).strip().upper()
    reasons: list[str] = []
    if not candidate_name:
        reasons.append("missing_candidate")
    if normalized_signal not in VALID_SIGNALS:
        reasons.append("invalid_signal")
    if timestamp is None or not str(timestamp).strip():
        reasons.append("missing_timestamp")
    else:
        try:
            _timestamp_key(timestamp)
        except (TypeError, ValueError, OverflowError):
            reasons.append("invalid_timestamp")
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


def compare_ab_by_timestamp(
    evaluation_timestamps: Iterable[object],
    baseline_trades: Iterable[object],
    candidate_trades: Iterable[object],
    *,
    min_samples: int = 100,
) -> dict:
    """Compare outcomes over the same explicit evaluation timestamps.

    Each timestamp is one evaluation slot (normally each OOS bar). A strategy
    with no trade at a slot receives 0R for that slot. Trade objects may expose
    timestamp and r attributes or provide those keys in a mapping. Duplicate
    slots, duplicate trades per slot, non-finite returns, and trades outside
    the supplied window fail closed. Expectancy is per slot, not per trade.
    """
    slots = [_timestamp_key(value) for value in evaluation_timestamps]
    if not slots:
        raise ValueError("evaluation_timestamps must not be empty")
    if len(set(slots)) != len(slots):
        raise ValueError("evaluation_timestamps must be unique")
    slot_set = set(slots)

    def read_trade(trade: object) -> tuple[str, float]:
        if isinstance(trade, dict):
            timestamp, outcome = trade.get("timestamp"), trade.get("r")
        else:
            timestamp, outcome = getattr(trade, "timestamp", None), getattr(trade, "r", None)
        if timestamp is None or outcome is None:
            raise ValueError("each trade must provide timestamp and r")
        key = _timestamp_key(timestamp)
        value = float(outcome)
        if not isfinite(value):
            raise ValueError("trade outcomes must be finite numbers")
        if key not in slot_set:
            raise ValueError("trade timestamp is outside evaluation window")
        return key, value

    def align(trades: Iterable[object]) -> list[float]:
        by_slot: dict[str, float] = {}
        for trade in trades:
            key, value = read_trade(trade)
            if key in by_slot:
                raise ValueError("at most one trade per evaluation timestamp is supported")
            by_slot[key] = value
        return [by_slot.get(slot, 0.0) for slot in slots]

    result = compare_ab(align(baseline_trades), align(candidate_trades), min_samples=min_samples)
    result["sample_unit"] = "evaluation_timestamp"
    result["no_trade_outcome_r"] = 0.0
    result["note"] = "Expectancy is per evaluation timestamp; not per-trade expectancy."
    return result
