from __future__ import annotations

import itertools
import random
from dataclasses import asdict
from typing import Iterable

import numpy as np
import pandas as pd

from backtest.engine import run_backtest, run_backtest_window, _run_backtest_indicators
from backtest.metrics import metrics
from strategy.strategy import StrategyParams, indicators


SELECTION_SPREAD = 0.00005
# Keep the expensive chronological stability pass focused on the strongest
# training candidates. This changes no OOS data usage: all screening remains
# inside the training window.
MAX_STABILITY_CANDIDATES = 20


def _score(m):
    pf = -1.0 if m["profit_factor"] is None else m["profit_factor"]
    return (m["expectancy"], pf, -m["max_drawdown"], m["trades"])


def _stability_score(tune: pd.DataFrame, p: StrategyParams, spread: float = SELECTION_SPREAD, precomputed=None):
    """Score a candidate on chronological slices with conservative spread costs."""
    if len(tune) < 60:
        m = metrics([t.r for t in (_run_backtest_indicators(precomputed, p, spread=spread) if precomputed is not None else run_backtest(tune, p, spread=spread))])
        return _score(m), [m]
    index_chunks = np.array_split(np.arange(len(tune)), 3)
    slice_metrics = []
    for idx in index_chunks:
        if precomputed is not None:
            tx = precomputed.iloc[idx].reset_index(drop=True)
            trades = _run_backtest_indicators(tx, p, spread=spread)
        else:
            trades = run_backtest(tune.iloc[idx].reset_index(drop=True), p, spread=spread)
        slice_metrics.append(metrics([t.r for t in trades]))
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
    for ef, es, rp, am, rr, bm, strength, rlo, rhi in itertools.product(
        (15, 20, 25), (45, 50, 55), (14,), (1.25, 1.5), (1.5, 2.0), (0.45, 0.55),
        (0.15, 0.20, 0.25), (50.0, 52.0), (65.0, 68.0),
    ):
        if ef >= es:
            continue
        yield StrategyParams(ema_fast=ef, ema_slow=es, rsi_period=rp,
                             atr_period=14, atr_multiplier=am, rr=rr,
                             body_min=bm, hypothesis="trend_filtered", trend_strength=strength, rsi_low=rlo, rsi_high=rhi)


def mean_reversion_candidates():
    for rp, am, rr, bm in itertools.product(
        (10, 14), (1.25, 1.5), (1.5, 2.0), (0.45, 0.55),
    ):
        yield StrategyParams(ema_fast=20, ema_slow=50, rsi_period=rp,
                             atr_period=14, atr_multiplier=am, rr=rr,
                             body_min=bm, hypothesis="mean_reversion")


def breakout_candidates():
    for dp, am, rr, bm in itertools.product(
        (10, 15, 20, 30), (1.25, 1.5), (1.5, 2.0), (0.40, 0.50),
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
    indicator_cache = {}

    # Stage 1: cheap fit screening for every candidate. Indicators are cached
    # by period parameters so repeated combinations reuse rolling calculations.
    for p in candidates:
        key = (p.ema_fast, p.ema_slow, p.rsi_period, p.atr_period, p.donchian_period)
        x_fit = indicator_cache.get(key)
        if x_fit is None:
            x_fit = indicators(fit, p)
            indicator_cache[key] = x_fit
        fit_trades = _run_backtest_indicators(x_fit, p, spread=selection_spread)
        fit_m = metrics([t.r for t in fit_trades])
        if fit_m["trades"] >= min_trades:
            ranked.append((p, fit_m, x_fit))

    # Stage 2: the expensive chronological stability test is run only on the
    # strongest training candidates. No holdout/OOS observations are touched.
    ranked.sort(key=lambda x: _score(x[1]), reverse=True)
    stability_candidates = ranked[:MAX_STABILITY_CANDIDATES]

    stable = []
    for p, fit_m, x_fit in stability_candidates:
        stability, slice_metrics = _stability_score(
            fit, p, spread=selection_spread, precomputed=x_fit
        )
        if stability[0] > -900:
            stable.append((p, fit_m, stability, slice_metrics))

    best = baseline
    best_stability = None
    if stable:
        stable.sort(
            key=lambda x: (
                x[2][3], x[2][1], x[2][0], x[2][2],
                -x[1]["max_drawdown"], x[2][4]
            ),
            reverse=True,
        )
        best, _, best_stability, _ = stable[0]

    best_key = (
        best.ema_fast, best.ema_slow, best.rsi_period,
        best.atr_period, best.donchian_period
    )
    best_x_fit = indicator_cache.get(best_key)
    if best_x_fit is None:
        best_x_fit = indicators(fit, best)
    fit_m = metrics([
        t.r for t in _run_backtest_indicators(
            best_x_fit, best, spread=selection_spread
        )
    ])
    tune_m = metrics([
        t.r for t in run_backtest(tune, best, spread=selection_spread)
    ])
    return best, {
        "candidate_count": len(candidates),
        "eligible_count": len(ranked),
        "stability_evaluated": len(stability_candidates),
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
        oos_trades = run_backtest_window(df, selected, oos_start, oos_end)
        oos_m = metrics([t.r for t in oos_trades])
        baseline_trades = run_backtest_window(df, StrategyParams(hypothesis=hypothesis), oos_start, oos_end)
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
        neighbors = [metrics([t.r for t in run_backtest_window(df, q, start, end)]) for q in _neighbor_params(p)]
        base = metrics([t.r for t in run_backtest_window(df, p, start, end)])
        positive = sum(m["total_r"] > 0 for m in neighbors)
        rows.append({"fold": f["fold"], "selected_oos": base,
                     "neighbor_count": len(neighbors),
                     "neighbor_positive_count": positive,
                     "neighbor_positive_rate": positive / len(neighbors) if neighbors else 0.0})
        for spread in (0.0, 0.00005, 0.00010):
            spread_results.append({"fold": f["fold"], "spread": spread,
                                   "metrics": metrics([t.r for t in run_backtest_window(df, p, start, end, spread=spread)])})
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


def independent_holdout(df: pd.DataFrame, holdout_ratio: float = 0.20):
    """Evaluate each hypothesis once on a final unseen chronological holdout.

    Parameter selection is performed only on the pre-holdout portion. The
    holdout is never used for parameter or hypothesis selection.
    """
    if not 0.10 <= holdout_ratio <= 0.40:
        raise ValueError("holdout_ratio must be between 0.10 and 0.40")
    n = len(df)
    holdout_start = int(n * (1.0 - holdout_ratio))
    if holdout_start < 100 or n - holdout_start < 50:
        raise ValueError("dataset is too small for independent holdout")
    results = {}
    for hypothesis in ("trend_filtered", "trend", "mean_reversion", "breakout", "pullback"):
        print(f"FORWARD_HOLDOUT_HYPOTHESIS_START={hypothesis}", flush=True)
        selected, selection = select_params(
            df.iloc[:holdout_start].reset_index(drop=True),
            hypothesis=hypothesis,
        )
        trades = run_backtest_window(df, selected, holdout_start, n)
        holdout_metrics = metrics([t.r for t in trades])
        print(f"FORWARD_HOLDOUT_HYPOTHESIS_COMPLETE={hypothesis} trades={holdout_metrics['trades']} total_r={holdout_metrics['total_r']}", flush=True)
        results[hypothesis] = {
            "holdout_start": holdout_start,
            "holdout_rows": n - holdout_start,
            "params": asdict(selected),
            "selection": selection,
            "holdout_metrics": holdout_metrics,
        }
    return results
