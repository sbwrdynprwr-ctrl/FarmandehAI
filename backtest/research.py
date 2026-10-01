from __future__ import annotations

import itertools
import random
from dataclasses import asdict
from typing import Iterable

import numpy as np
import pandas as pd

from backtest.engine import run_backtest, run_backtest_window
from backtest.metrics import metrics
from strategy.strategy import StrategyParams


SELECTION_SPREAD = 0.00005


def _score(m):
    pf = -1.0 if m["profit_factor"] is None else m["profit_factor"]
    return (m["expectancy"], pf, -m["max_drawdown"], m["trades"])


def _stability_score(tune: pd.DataFrame, p: StrategyParams, spread: float = SELECTION_SPREAD):
    """Score a candidate on chronological slices with conservative spread costs."""
    if len(tune) < 60:
        m = metrics([t.r for t in run_backtest(tune, p, spread=spread)])
        return _score(m), [m]
    # Split by positional indices so numpy never coerces DataFrames to ndarrays.
    index_chunks = np.array_split(np.arange(len(tune)), 3)
    slice_metrics = [
        metrics([
            t.r
            for t in run_backtest(
                tune.iloc[idx].reset_index(drop=True),
                p,
                spread=spread,
            )
        ])
        for idx in index_chunks
    ]
    valid = [m for m in slice_metrics if m["trades"] >= 3]
    if len(valid) < 2:
        return (-999.0, -999.0, 999.0, 0, 0), slice_metrics
    mean_exp = float(np.mean([m["expectancy"] for m in valid]))
    min_exp = float(min(m["expectancy"] for m in valid))
    positive_slices = sum(m["total_r"] > 0 for m in valid)
    pf_values = [m["profit_factor"] for m in valid if m["profit_factor"] is not None]
    mean_pf = float(np.mean(pf_values)) if pf_values else -1.0
    total_trades = sum(m["trades"] for m in valid)
    return (mean_exp, min_exp, mean_pf, positive_slices, total_trades), slice_metrics


def parameter_candidates():
    for ef, es, rp, am, rr, bm in itertools.product(
        (20, 25), (45, 50, 55), (14,), (1.25, 1.5), (1.5, 2.0), (0.55,),
    ):
        if ef >= es:
            continue
        yield StrategyParams(ema_fast=ef, ema_slow=es, rsi_period=rp,
                             atr_period=14, atr_multiplier=am, rr=rr,
                             body_min=bm, hypothesis="trend")


def mean_reversion_candidates():
    for rp, am, rr, bm in itertools.product(
        (10, 14), (1.25, 1.5), (1.5, 2.0), (0.45, 0.55),
    ):
        yield StrategyParams(ema_fast=20, ema_slow=50, rsi_period=rp,
                             atr_period=14, atr_multiplier=am, rr=rr,
                             body_min=bm, hypothesis="mean_reversion")


def breakout_candidates():
    for dp, am, rr, bm in itertools.product(
        (15, 20, 30, 40), (1.25, 1.5), (1.5, 2.0), (0.45, 0.55),
    ):
        yield StrategyParams(ema_fast=20, ema_slow=50, rsi_period=14,
                             atr_period=14, atr_multiplier=am, rr=rr,
                             body_min=bm, hypothesis="breakout",
                             donchian_period=dp)


def pullback_candidates():
    for ef, es, am, rr, bm in itertools.product(
        (15, 20, 25), (45, 50, 55), (1.25, 1.5), (1.5, 2.0), (0.45, 0.55),
    ):
        if ef >= es:
            continue
        yield StrategyParams(ema_fast=ef, ema_slow=es, rsi_period=14,
                             atr_period=14, atr_multiplier=am, rr=rr,
                             body_min=bm, hypothesis="pullback")


def _candidates_for(hypothesis):
    if hypothesis == "mean_reversion":
        return list(mean_reversion_candidates())
    if hypothesis == "breakout":
        return list(breakout_candidates())
    if hypothesis == "pullback":
        return list(pullback_candidates())
    return list(parameter_candidates())


def select_params(
    train: pd.DataFrame,
    min_trades: int = 10,
    hypothesis: str = "trend",
    selection_spread: float = SELECTION_SPREAD,
):
    split = max(1, int(len(train) * 0.75))
    fit = train.iloc[:split].reset_index(drop=True)
    tune = train.iloc[split:].reset_index(drop=True)
    candidates = _candidates_for(hypothesis)
    baseline = StrategyParams(hypothesis=hypothesis)
    ranked = []
    for p in candidates:
        fit_trades = run_backtest(fit, p, spread=selection_spread)
        fit_m = metrics([t.r for t in fit_trades])
        if fit_m["trades"] >= min_trades:
            stability, slice_metrics = _stability_score(
                fit, p, spread=selection_spread
            )
            if stability[0] > -900:
                ranked.append((p, fit_m, stability, slice_metrics))
    best = baseline
    best_stability = None
    if ranked:
        ranked.sort(key=lambda x: (x[2][3], x[2][1], x[2][0], x[2][2], -x[1]["max_drawdown"], x[2][4]), reverse=True)
        best, _, best_stability, _ = ranked[0]
    fit_m = metrics([t.r for t in run_backtest(fit, best, spread=selection_spread)])
    tune_m = metrics([t.r for t in run_backtest(tune, best, spread=selection_spread)])
    return best, {
        "candidate_count": len(candidates),
        "eligible_count": len(ranked),
        "fit_metrics": fit_m,
        "tune_metrics": tune_m,
        "stability_score": best_stability,
        "selection_score": _score(tune_m),
        "selection_spread": selection_spread,
    }


def walk_forward_search(df: pd.DataFrame, folds: int = 4, train_ratio: float = 0.5, hypothesis: str = "trend"):
    if folds < 2:
        raise ValueError("folds must be >= 2")
    n = len(df)
    first_oos = max(1, int(n * train_ratio))
    remaining = n - first_oos
