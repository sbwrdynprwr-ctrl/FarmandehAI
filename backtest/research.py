from __future__ import annotations

import itertools
import random
from dataclasses import asdict
from typing import Iterable

import numpy as np
import pandas as pd

from backtest.engine import run_backtest
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
    cuts = np.array_split(tune, 3)
    slice_metrics = [
        metrics([t.r for t in run_backtest(part.reset_index(drop=True), p, spread=spread)])
        for part in cuts
    ]
    # Require enough activity before treating a slice as evidence.
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
    # Selection remains entirely inside the training window. OOS is never used.
    split = max(1, int(len(train) * 0.75))
    fit = train.iloc[:split].reset_index(drop=True)
    tune = train.iloc[split:].reset_index(drop=True)
    candidates = _candidates_for(hypothesis)
    baseline = StrategyParams(hypothesis=hypothesis)
    ranked = []
    for p in candidates:
        # Candidate eligibility and ranking use FIT only. The tune slice is
        # held out inside the training window and is never used to choose p.
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
        ranked.sort(key=lambda x: x[2], reverse=True)
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
    oos_size = remaining // folds
    if oos_size < 20:
        raise ValueError("OOS fold is too small")
    folds_out, combined = [], []
    for k in range(folds):
        print(f"RESEARCH_FOLD_START={k + 1}/{folds} HYPOTHESIS={hypothesis}", flush=True)
        oos_start = first_oos + k * oos_size
        oos_end = first_oos + (k + 1) * oos_size if k < folds - 1 else n
        train = df.iloc[:oos_start].reset_index(drop=True)
        oos = df.iloc[oos_start:oos_end].reset_index(drop=True)
        selected, selection = select_params(train, hypothesis=hypothesis)
        oos_trades = run_backtest(oos, selected)
        oos_m = metrics([t.r for t in oos_trades])
        baseline_trades = run_backtest(oos, StrategyParams(hypothesis=hypothesis))
        baseline_m = metrics([t.r for t in baseline_trades])
        folds_out.append({
            "fold": k + 1, "train_rows": len(train), "oos_rows": len(oos),
            "params": asdict(selected), "selection": selection,
            "oos_metrics": oos_m, "baseline_oos_metrics": baseline_m,
            "trades": oos_trades,
        })
        combined.extend(oos_trades)
        print(f"RESEARCH_FOLD_COMPLETE={k + 1}/{folds} HYPOTHESIS={hypothesis} trades={oos_m['trades']} total_r={oos_m['total_r']}", flush=True)
    return folds_out, metrics([t.r for t in combined])


def _neighbor_params(p: StrategyParams):
    variants = []
    grids = {
        "ema_fast": (max(5, p.ema_fast - 5), p.ema_fast + 5),
        "ema_slow": (max(p.ema_fast + 5, p.ema_slow - 10), p.ema_slow + 10),
        "rsi_period": (max(5, p.rsi_period - 2), p.rsi_period + 2),
        "atr_multiplier": (max(0.5, p.atr_multiplier - 0.25), p.atr_multiplier + 0.25),
        "rr": (max(1.0, p.rr - 0.5), p.rr + 0.5),
        "body_min": (max(0.3, p.body_min - 0.05), min(0.9, p.body_min + 0.05)),
    }
    if p.hypothesis == "breakout":
        grids["donchian_period"] = (max(5, p.donchian_period - 5), p.donchian_period + 5)
    for name, vals in grids.items():
        for v in vals:
            d = asdict(p)
            d[name] = v
            if d["ema_fast"] < d["ema_slow"]:
                variants.append(StrategyParams(**d))
    return variants


def robustness(df: pd.DataFrame, folds_out):
    rows, spread_results = [], []
    first_oos = int(len(df) * 0.5)
    oos_size = (len(df) - first_oos) // max(1, len(folds_out))
    for f in folds_out:
        start = first_oos + (f["fold"] - 1) * oos_size
        end = first_oos + f["fold"] * oos_size
        if f["fold"] == len(folds_out):
            end = len(df)
        oos = df.iloc[start:end].reset_index(drop=True)
        p = StrategyParams(**f["params"])
        neighbors = [metrics([t.r for t in run_backtest(oos, q)]) for q in _neighbor_params(p)]
        base = metrics([t.r for t in run_backtest(oos, p)])
        positive = sum(m["total_r"] > 0 for m in neighbors)
        rows.append({"fold": f["fold"], "selected_oos": base,
                     "neighbor_count": len(neighbors),
                     "neighbor_positive_count": positive,
                     "neighbor_positive_rate": positive / len(neighbors) if neighbors else 0.0})
        for spread in (0.0, 0.00005, 0.00010):
            spread_results.append({"fold": f["fold"], "spread": spread,
                                   "metrics": metrics([t.r for t in run_backtest(oos, p, spread=spread)])})
    return {"folds": rows, "spread_sensitivity": spread_results}


def monte_carlo(rs: Iterable[float], simulations: int = 1000, seed: int = 42):
    vals = list(map(float, rs))
    if not vals:
        return {"simulations": 0, "max_dd_p50": 0.0, "max_dd_p95": 0.0, "terminal_r": 0.0}
    rng, dds = random.Random(seed), []
    for _ in range(simulations):
        x = vals[:]
        rng.shuffle(x)
        equity = np.cumsum(x)
        peak = np.maximum.accumulate(np.r_[0.0, equity])
        dds.append(float((peak[1:] - equity).max() if len(equity) else 0.0))
    return {"simulations": simulations, "max_dd_p50": float(np.percentile(dds, 50)),
            "max_dd_p95": float(np.percentile(dds, 95)), "max_dd_max": float(max(dds)),
            "terminal_r": float(sum(vals))}
