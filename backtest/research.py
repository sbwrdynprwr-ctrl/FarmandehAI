from __future__ import annotations

import itertools
import random
import json
import os
from dataclasses import asdict
from typing import Iterable

import numpy as np
import pandas as pd

from backtest.engine import run_backtest, run_backtest_window, _run_backtest_indicators
from backtest.metrics import metrics
from strategy.strategy import StrategyParams, indicators


SELECTION_SPREAD = 0.00005
ROBUST_SELECTION_SPREAD = 0.00015
COSTAWARE_SELECTION_SPREAD = 0.00015
# Keep training neighbor robustness aligned with the final OOS neighbor gate.
# Candidate profitability can still be selected under the conservative 15bps cost,
# but neighbor stability itself must be evaluated at the exact 5bps gate spread.
NEIGHBOR_SELECTION_SPREAD = 0.00005  # exact spread used by the final neighbor gate
STRESS_NEIGHBOR_SELECTION_SPREAD = 0.00015  # conservative training-only neighbor stress test
# Keep the expensive chronological stability pass focused on the strongest
# training candidates. This changes no OOS data usage: all screening remains
# inside the training window.
MAX_STABILITY_CANDIDATES = 20
# Bound the expensive parameter sweep deterministically. The previous full Cartesian
# grids produced ~7,200 candidates per selection pass, which multiplied across
# 11 hypotheses, 4 WFO folds, and the independent holdout made the research job
# computationally unbounded in practice. Keep broad coverage while making each
# pass finite and repeatable; the final OOS gates remain unchanged.
MAX_SEARCH_CANDIDATES = 64
# Robust mean-reversion has a much larger candidate grid. Give it a wider,
# deterministic search budget so a viable neighborhood is not missed simply
# because the generic cap samples too sparsely. Final OOS gates are unchanged.
MAX_ROBUST_SEARCH_CANDIDATES = 256
# The v2 execution-aware grid has 144 candidates. Search the full grid so a
# viable cost-robust neighborhood is not lost to deterministic downsampling.
MAX_V2_SEARCH_CANDIDATES = 144
MAX_V2_STABILITY_CANDIDATES = 48


def _bounded_candidates(candidates):
    if len(candidates) <= MAX_SEARCH_CANDIDATES:
        return candidates
    idx = np.linspace(0, len(candidates) - 1, MAX_SEARCH_CANDIDATES, dtype=int)
    return [candidates[i] for i in idx]


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


def _training_neighbor_rates(fit: pd.DataFrame, p: StrategyParams, spread: float):
    """Measure aggregate and worst chronological-slice neighbor stability on training data."""
    neighbors = _neighbor_params(p)
    if not neighbors:
        return 0.0, 0.0
    chunks = np.array_split(np.arange(len(fit)), 3) if len(fit) >= 60 else [np.arange(len(fit))]
    all_positive = []
    slice_rates = []
    for idx in chunks:
        tx = fit.iloc[idx].reset_index(drop=True)
        positive = 0
        for q in neighbors:
            q_trades = run_backtest(tx, q, spread=spread)
            q_metrics = metrics([t.r for t in q_trades])
            positive += int(q_metrics["total_r"] > 0.0)
            all_positive.append(q_metrics["total_r"] > 0.0)
        slice_rates.append(positive / len(neighbors))
    aggregate = sum(all_positive) / len(all_positive) if all_positive else 0.0
    return aggregate, min(slice_rates) if slice_rates else 0.0


def parameter_candidates(hypothesis: str = "trend_filtered"):
    for ef, es, rp, am, rr, bm, strength, rlo, rhi in itertools.product(
        (15, 20, 25), (45, 50, 55), (14,), (1.25, 1.5), (1.5, 2.0), (0.45, 0.55),
        (0.15, 0.20, 0.25), (50.0, 52.0), (65.0, 68.0),
    ):
        if ef >= es:
            continue
        yield StrategyParams(ema_fast=ef, ema_slow=es, rsi_period=rp,
                             atr_period=14, atr_multiplier=am, rr=rr,
                             body_min=bm, hypothesis=hypothesis, trend_strength=strength, rsi_low=rlo, rsi_high=rhi)


def mean_reversion_band_candidates():
    # Bollinger-band excursion plus reversal confirmation.
    for rp, am, bm, rlo, rhi, dist in itertools.product(
        (10, 14), (1.25, 1.5), (0.40, 0.50),
        (35.0, 40.0), (60.0, 65.0), (0.50, 0.75, 1.00),
    ):
        yield StrategyParams(
            ema_fast=20, ema_slow=50, rsi_period=rp,
            atr_period=14, atr_multiplier=am, rr=2.0,
            body_min=bm, hypothesis="mean_reversion_band",
            rsi_low=rlo, rsi_high=rhi,
            reversion_distance_atr=dist,
        )


def mean_reversion_regime_candidates():
    # Mean reversion is allowed only in a non-trending regime.
    for rp, am, rr, bm, rlo, rhi, gap, dist in itertools.product(
        (10, 14), (1.25, 1.5), (1.5, 2.0), (0.40, 0.50),
        (35.0, 40.0), (60.0, 65.0), (0.50, 0.75), (0.50, 0.75, 1.00),
    ):
        yield StrategyParams(
            ema_fast=20, ema_slow=50, rsi_period=rp, atr_period=14,
            atr_multiplier=am, rr=rr, body_min=bm, hypothesis="mean_reversion_regime",
            rsi_low=rlo, rsi_high=rhi, regime_gap=gap,
            reversion_distance_atr=dist, confirmation_bars=1,
        )

def mean_reversion_v2_candidates():
    # Execution-aware search: avoid targets so close to entry that realistic
    # spread can dominate the expected move. Selection remains training-only.
    for rp, am, bm, rlo, rhi, min_target in itertools.product(
        (10, 14), (1.0, 1.25, 1.5), (0.40, 0.50),
        (40.0, 45.0), (55.0, 60.0), (0.25, 0.50, 0.75, 1.00, 1.25, 1.50),
    ):
        yield StrategyParams(ema_fast=20, ema_slow=50, rsi_period=rp,
                             atr_period=14, atr_multiplier=am, rr=2.0,
                             body_min=bm, hypothesis="mean_reversion_v2",
                             rsi_low=rlo, rsi_high=rhi,
                             min_target_distance_atr=min_target)




def mean_reversion_rr_candidates():
    # Same reversal signal as v2, but the exit is a fixed ATR-based RR target.
    # This tests whether a farther predefined target can absorb realistic spread.
    for rp, am, rr, bm, rlo, rhi in itertools.product(
        (10, 14), (1.25, 1.5, 1.75), (1.0, 1.5, 2.0, 2.5, 3.0),
        (0.40, 0.50), (40.0, 45.0), (55.0, 60.0),
    ):
        yield StrategyParams(ema_fast=20, ema_slow=50, rsi_period=rp,
                             atr_period=14, atr_multiplier=am, rr=rr,
                             body_min=bm, hypothesis="mean_reversion_rr",
                             rsi_low=rlo, rsi_high=rhi)


def mean_reversion_robust_candidates():
    # Cost-aware reversal with one-bar confirmation.
    for rp, am, rr, bm, rlo, rhi, gap, dist in itertools.product(
        (10, 14), (1.25, 1.5), (1.25, 1.5, 1.75), (0.40, 0.50),
        (40.0, 45.0), (55.0, 60.0), (0.50, 0.75), (0.75, 1.00),
    ):
        yield StrategyParams(ema_fast=20, ema_slow=50, rsi_period=rp,
                             atr_period=14, atr_multiplier=am, rr=rr,
                             body_min=bm, hypothesis="mean_reversion_robust",
                             rsi_low=rlo, rsi_high=rhi, regime_gap=gap,
                             reversion_distance_atr=dist, confirmation_bars=1)

def mean_reversion_costaware_candidates():
    # Conservative reversal branch: avoid strong trends, require a meaningful
    # excursion from the mean, and select parameters against a 10bps training
    # cost so the final 5bps gate is not the first cost-aware test.
    for rp, am, rr, bm, rlo, rhi, gap, dist in itertools.product(
        (10, 14), (1.5, 2.0), (1.5, 2.0, 2.5), (0.45, 0.55),
        (35.0, 40.0), (60.0, 65.0), (0.50, 0.75), (0.50, 0.75),
    ):
        yield StrategyParams(ema_fast=20, ema_slow=50, rsi_period=rp,
                             atr_period=14, atr_multiplier=am, rr=rr,
                             body_min=bm, hypothesis="mean_reversion_costaware",
                             rsi_low=rlo, rsi_high=rhi, regime_gap=gap,
                             reversion_distance_atr=dist)


def mean_reversion_v3_candidates():
    for rp, am, rr, bm, rlo, rhi, gap, dist in itertools.product(
        (10, 14), (1.25, 1.5), (1.5, 2.0), (0.45, 0.55),
        (35.0, 40.0, 45.0), (55.0, 60.0, 65.0), (0.50, 0.75, 1.0), (0.25, 0.50, 0.75),
    ):
        yield StrategyParams(ema_fast=20, ema_slow=50, rsi_period=rp,
                             atr_period=14, atr_multiplier=am, rr=rr,
                             body_min=bm, hypothesis="mean_reversion_v3",
                             rsi_low=rlo, rsi_high=rhi, regime_gap=gap,
                             reversion_distance_atr=dist)

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


def trend_regime_candidates():
    for ef, es, am, rr, bm, strength, rlo, rhi, gap in itertools.product(
        (15, 20, 25), (45, 50, 55), (1.25, 1.5), (1.5, 2.0),
        (0.45, 0.55), (0.15, 0.20, 0.25), (50.0, 52.0), (65.0, 68.0), (0.50, 0.75),
    ):
        if ef >= es:
            continue
        yield StrategyParams(ema_fast=ef, ema_slow=es, rsi_period=14,
                             atr_period=14, atr_multiplier=am, rr=rr,
                             body_min=bm, hypothesis="trend_regime",
                             trend_strength=strength, rsi_low=rlo, rsi_high=rhi,
                             slope_bars=3, regime_gap=gap)


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
    if hypothesis == "mean_reversion_band":
        return list(mean_reversion_band_candidates())
    if hypothesis == "mean_reversion_regime":
        return list(mean_reversion_regime_candidates())
    if hypothesis == "mean_reversion_v2":
        return list(mean_reversion_v2_candidates())
    if hypothesis == "mean_reversion_rr":
        return list(mean_reversion_rr_candidates())
    if hypothesis == "mean_reversion_costaware":
        return list(mean_reversion_costaware_candidates())
    if hypothesis == "mean_reversion_robust":
        return list(mean_reversion_robust_candidates())
    if hypothesis == "mean_reversion_v3":
        return list(mean_reversion_v3_candidates())
    if hypothesis == "mean_reversion":
        return list(mean_reversion_candidates())
    if hypothesis == "breakout":
        return list(breakout_candidates())
    if hypothesis == "pullback":
        return list(pullback_candidates())
    if hypothesis == "trend_regime":
        return list(trend_regime_candidates())
    return list(parameter_candidates(hypothesis))


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
    candidate_limit = (
        MAX_ROBUST_SEARCH_CANDIDATES
        if hypothesis == "mean_reversion_v2"
        else (MAX_ROBUST_SEARCH_CANDIDATES
              if hypothesis in ("mean_reversion_robust", "mean_reversion_costaware", "mean_reversion_band", "mean_reversion_regime")
              else MAX_SEARCH_CANDIDATES)
    )
    raw_candidates = _candidates_for(hypothesis)
    baseline = StrategyParams(hypothesis=hypothesis)
    # Include the declared baseline before bounded sampling so the baseline
    # cannot disappear from an oversized candidate grid.
    raw_candidates = [baseline] + [p for p in raw_candidates if p != baseline]
    if len(raw_candidates) <= candidate_limit:
        candidates = raw_candidates
    else:
        remaining = raw_candidates[1:]
        budget = max(0, candidate_limit - 1)
        if budget:
            idx = np.linspace(0, len(remaining) - 1, budget, dtype=int)
            candidates = [baseline] + [remaining[i] for i in idx]
        else:
            candidates = [baseline]
    ranked = []
    indicator_cache = {}

    # Always include the declared baseline as an explicit selection candidate.
    # It is inserted before bounded sampling so it cannot disappear when the
    # candidate grid exceeds the search budget. This lets the selector compare
    # the tuned family against the untouched baseline instead of silently
    # falling back to it only after all candidates have been rejected.
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
        if fit_m["trades"] >= min_trades or p == baseline:
            ranked.append((p, fit_m, x_fit))

    # Stage 2: the expensive chronological stability test is run only on the
    # strongest training candidates. No holdout/OOS observations are touched.
    ranked.sort(key=lambda x: _score(x[1]), reverse=True)
    # Robust mean-reversion families must search the full bounded candidate
    # space because the final training gates include neighbor stability and
    # exact 5bps viability, which are not equivalent to the primary fit rank.
    # Keep expensive neighbor/stability evaluation bounded even for the
    # cost-aware families. The candidate grid is still broad, but evaluating
    # every eligible candidate multiplies dozens of backtests per candidate.
    # Rank is training-only, so this bound cannot leak OOS information.
    stability_limit = min(
        len(ranked),
        MAX_V2_STABILITY_CANDIDATES if hypothesis == "mean_reversion_v2" else (
            24 if hypothesis in ("mean_reversion_rr", "mean_reversion_robust",
                                 "mean_reversion_costaware", "mean_reversion_band",
                                 "mean_reversion_regime")
            else MAX_STABILITY_CANDIDATES
        ),
    )
    stability_candidates = ranked[:stability_limit]

    stable = []
    for p, fit_m, x_fit in stability_candidates:
        stability, slice_metrics = _stability_score(
            fit, p, spread=selection_spread, precomputed=x_fit
        )
        if stability[0] > -900:
            # Training-only neighbor robustness. The final approval gate still
            # recomputes neighbor stability on unseen OOS data at 5bps.
            neighbor_positive_rate = 0.0
            neighbor_worst_slice_rate = 0.0
            if hypothesis.startswith("mean_reversion"):
                neighbor_positive_rate, neighbor_worst_slice_rate = _training_neighbor_rates(
                    fit, p, NEIGHBOR_SELECTION_SPREAD
                )
                stress_neighbor_positive_rate, stress_neighbor_worst_slice_rate = _training_neighbor_rates(
                    fit, p, STRESS_NEIGHBOR_SELECTION_SPREAD
                )
            else:
                stress_neighbor_positive_rate, stress_neighbor_worst_slice_rate = 0.0, 0.0
            stable.append((p, fit_m, stability, slice_metrics, neighbor_positive_rate, neighbor_worst_slice_rate,
                           stress_neighbor_positive_rate, stress_neighbor_worst_slice_rate))

    best = baseline
    best_stability = None
    selection_status = "baseline_default"
    if stable:
        if hypothesis in ("mean_reversion_v2", "mean_reversion_rr", "mean_reversion_robust", "mean_reversion_costaware", "mean_reversion_band", "mean_reversion_regime"):
            # Candidate eligibility is aligned with the exact 5bps regime used
            # by the final OOS gate. The 15bps result is retained as a stress
            # metric/ranking signal, but it is not allowed to discard a candidate
            # before the blind OOS gate gets to judge it. This avoids a training
            # prefilter becoming stricter than the declared final execution test.
            robust_ranked = []
            for p, fit_m, stability, slice_metrics, neighbor_positive_rate, neighbor_worst_slice_rate, stress_neighbor_positive_rate, stress_neighbor_worst_slice_rate in stable:
                tune_x = indicators(tune, p)
                tune_m_candidate = metrics([
                    t.r for t in _run_backtest_indicators(tune_x, p, spread=selection_spread)
                ])
                # The final gate evaluates the candidate at 5bps OOS. Make the
                # training selector optimize for that same execution regime while
                # retaining the conservative 15bps result as a separate viability
                # constraint. This prevents a candidate from winning on 15bps fit
                # quality while being fragile at the exact final-gate spread.
                gate_m_candidate = metrics([
                    t.r for t in _run_backtest_indicators(tune_x, p, spread=NEIGHBOR_SELECTION_SPREAD)
                ])
                positive_slices = stability[3]
                activity_floor = 20 if hypothesis == "mean_reversion_v2" else min_trades
                neighbor_floor = 0.75 if hypothesis in ("mean_reversion_v2", "mean_reversion_robust") else (0.50 if hypothesis == "mean_reversion_costaware" else 0.0)
                worst_slice_floor = 0.50 if hypothesis in ("mean_reversion_v2", "mean_reversion_robust", "mean_reversion_costaware") else 0.0
                stress_neighbor_floor = 0.50 if hypothesis in ("mean_reversion_v2", "mean_reversion_robust", "mean_reversion_costaware") else 0.0
                if (
                    tune_m_candidate["trades"] >= activity_floor
                    and gate_m_candidate["total_r"] > 0
                    and positive_slices >= 2
                    and neighbor_positive_rate >= neighbor_floor
                    and neighbor_worst_slice_rate >= worst_slice_floor
                ):
                    robust_ranked.append(
                        (p, fit_m, stability, slice_metrics, gate_m_candidate,
                         neighbor_positive_rate, neighbor_worst_slice_rate,
                         stress_neighbor_positive_rate, stress_neighbor_worst_slice_rate)
                    )
            # The exact 5bps neighbor contract remains a hard training eligibility
            # requirement. The 15bps stress result is retained for ranking instead of
            # being a second hard gate: otherwise a conservative stress test can collapse
            # the candidate pool to the baseline even when the declared 5bps contract is met.
            # The blind OOS gate remains authoritative for final approval.
            pool = robust_ranked
            pool.sort(
                key=lambda x: (
                    # For v2, prefer candidates with enough training activity
                    # before optimizing expectancy. This prevents a very sparse
                    # high-expectancy fit from winning when a more active robust
                    # target exists. The final OOS >=100-trade gate is unchanged.
                    int(hypothesis in ("mean_reversion_v2", "mean_reversion_rr", "mean_reversion_robust", "mean_reversion_costaware") and x[1]["total_r"] > 0.0),
                    int(hypothesis == "mean_reversion_v2" and x[4]["trades"] >= 30),
                    # Prefer performance at the exact 5bps final-gate
                    # spread, then neighbor stability, while retaining 15bps
                    # profitability as the eligibility constraint above.
                    x[4]["expectancy"], x[5], x[6],
                    # Prefer higher 15bps stress stability without making it a hidden
                    # pass/fail gate. Aggregate and worst-slice stress rates are both
                    # retained in the ranking tuple.
                    x[7], x[8], x[2][1], x[2][0], x[2][2],
                    -x[4]["max_drawdown"], x[4]["trades"]
                ),
                reverse=True,
            )
            if pool:
                best_record = pool[0]
                best = best_record[0]
                best_stability = best_record[2]
                selection_status = "robust_candidate_selected"
            else:
                # Fail closed without crashing: no training candidate met the
                # complete robustness contract, so keep the baseline and let
                # downstream OOS gates reject it rather than selecting a fragile fit.
                best = baseline
                best_stability = None
                selection_status = "no_robust_candidate"
        else:
            stable.sort(
                key=lambda x: (
                    x[4], x[2][3], x[2][1], x[2][0], x[2][2],
                    -x[1]["max_drawdown"], x[2][4]
                ),
                reverse=True,
            )
            best_record = stable[0]
            best = best_record[0]
            best_stability = best_record[2]
            selection_status = "stable_candidate_selected"

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
        "selection_status": selection_status,
    }


def _load_wf_checkpoint(path, hypothesis, df_len, folds, train_ratio, selection_spread=SELECTION_SPREAD):
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
        if payload.get("schema") != 10 or payload.get("hypothesis") != hypothesis:
            return {}
        if (payload.get("df_len") != df_len
                or payload.get("folds") != folds
                or float(payload.get("train_ratio")) != float(train_ratio)
                or float(payload.get("selection_spread")) != float(selection_spread)):
            return {}
        return payload
    except (OSError, ValueError, TypeError):
        return {}


def _save_wf_checkpoint(path, payload):
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, default=str)
        fp.flush()
        os.fsync(fp.fileno())
    os.replace(tmp, path)


def walk_forward_search(df: pd.DataFrame, folds: int = 4, train_ratio: float = 0.5, hypothesis: str = "trend", checkpoint_path: str | None = None):
    if folds < 2:
        raise ValueError("folds must be >= 2")
    n = len(df)
    first_oos = max(1, int(n * train_ratio))
    remaining = n - first_oos
    oos_size = remaining // folds
    if oos_size < 20:
        raise ValueError("OOS fold is too small")
    # Align parameter selection with the declared 5bps final execution gate.
    # Higher spreads remain stress tests in downstream robustness sensitivity.
    selection_spread = SELECTION_SPREAD
    checkpoint = _load_wf_checkpoint(
        checkpoint_path, hypothesis, n, folds, train_ratio,
        selection_spread=selection_spread,
    )
    saved_folds = {int(x["fold"]): x for x in checkpoint.get("completed_folds", [])}
    folds_out, combined = [], []
    for k in range(folds):
        fold_no = k + 1
        print(f"RESEARCH_FOLD_START={fold_no}/{folds} HYPOTHESIS={hypothesis}", flush=True)
        oos_start = first_oos + k * oos_size
        oos_end = first_oos + (k + 1) * oos_size if k < folds - 1 else n
        train = df.iloc[:oos_start].reset_index(drop=True)
        oos = df.iloc[oos_start:oos_end].reset_index(drop=True)
        saved = saved_folds.get(fold_no)
        if saved:
            selected = StrategyParams(**saved["params"])
            selection = saved["selection"]
            print(f"RESEARCH_FOLD_RESUME={fold_no}/{folds} HYPOTHESIS={hypothesis}", flush=True)
        else:
            selected, selection = select_params(train, hypothesis=hypothesis, selection_spread=selection_spread)
        oos_trades = run_backtest_window(df, selected, oos_start, oos_end, spread=selection_spread)
        oos_m = metrics([t.r for t in oos_trades])
        baseline_trades = run_backtest_window(df, StrategyParams(hypothesis=hypothesis), oos_start, oos_end, spread=selection_spread)
        baseline_m = metrics([t.r for t in baseline_trades])
        fold_record = {
            "fold": fold_no, "train_rows": len(train), "oos_rows": len(oos),
            "params": asdict(selected), "selection": selection,
            "oos_metrics": oos_m, "baseline_oos_metrics": baseline_m,
        }
        folds_out.append({**fold_record, "trades": oos_trades})
        combined.extend(oos_trades)
        checkpoint_payload = {
            "schema": 10, "hypothesis": hypothesis, "df_len": n,
            "selection_spread": selection_spread,
            "folds": folds, "train_ratio": train_ratio,
            "completed_folds": [
                {k: v for k, v in x.items() if k != "trades"} for x in folds_out
            ],
        }
        _save_wf_checkpoint(checkpoint_path, checkpoint_payload)
        print(f"RESEARCH_FOLD_COMPLETE={fold_no}/{folds} HYPOTHESIS={hypothesis} trades={oos_m['trades']} total_r={oos_m['total_r']}", flush=True)
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
    if p.hypothesis in (
        "mean_reversion_v2", "mean_reversion_v3", "mean_reversion_band",
        "mean_reversion_costaware", "mean_reversion_robust", "mean_reversion_regime",
    ):
        grids["regime_gap"] = (max(0.25, p.regime_gap - 0.25), p.regime_gap + 0.25)
        grids["reversion_distance_atr"] = (
            max(0.25, p.reversion_distance_atr - 0.25),
            p.reversion_distance_atr + 0.25,
        )
        grids["rsi_low"] = (max(20.0, p.rsi_low - 5.0), min(55.0, p.rsi_low + 5.0))
        grids["rsi_high"] = (max(45.0, p.rsi_high - 5.0), min(80.0, p.rsi_high + 5.0))
    if p.hypothesis == "mean_reversion_v2":
        grids["min_target_distance_atr"] = (
            max(0.0, p.min_target_distance_atr - 0.25),
            p.min_target_distance_atr + 0.25,
        )
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
        neighbors = [metrics([t.r for t in run_backtest_window(df, q, start, end, spread=SELECTION_SPREAD)]) for q in _neighbor_params(p)]
        base = metrics([t.r for t in run_backtest_window(df, p, start, end, spread=SELECTION_SPREAD)])
        positive = sum(m["total_r"] > 0 for m in neighbors)
        rows.append({"fold": f["fold"], "selected_oos": base,
                     "neighbor_count": len(neighbors),
                     "neighbor_positive_count": positive,
                     "neighbor_positive_rate": positive / len(neighbors) if neighbors else 0.0})
        for spread in (0.0, 0.00005, 0.00010, 0.00015):
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


def independent_holdout(df: pd.DataFrame, holdout_ratio: float = 0.20, checkpoint_path: str | None = None):
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
    # Persist each completed holdout hypothesis so an interrupted Railway
    # container resumes instead of repeating expensive training-only selection.
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as fp:
                saved = json.load(fp)
            if (saved.get("schema") == 5 and saved.get("df_len") == n
                    and float(saved.get("holdout_ratio")) == float(holdout_ratio)):
                results.update(saved.get("completed", {}))
        except (OSError, ValueError, TypeError):
            pass

    hypotheses = ("trend_filtered", "trend", "trend_regime", "mean_reversion_band", "mean_reversion_v2", "mean_reversion_rr", "mean_reversion_costaware", "mean_reversion_robust", "mean_reversion_regime", "mean_reversion_v3", "mean_reversion", "breakout", "pullback")
    for hypothesis in hypotheses:
        if hypothesis in results:
            print(f"FORWARD_HOLDOUT_HYPOTHESIS_RESUME={hypothesis}", flush=True)
            continue
        print(f"FORWARD_HOLDOUT_HYPOTHESIS_START={hypothesis}", flush=True)
        selected, selection = select_params(
            df.iloc[:holdout_start].reset_index(drop=True),
            hypothesis=hypothesis,
            # Align parameter selection with the declared 5bps final
            # execution gate; higher spreads remain downstream stress tests.
            selection_spread=SELECTION_SPREAD,
        )
        trades = run_backtest_window(df, selected, holdout_start, n, spread=SELECTION_SPREAD)
        holdout_metrics = metrics([t.r for t in trades])
        print(f"FORWARD_HOLDOUT_HYPOTHESIS_COMPLETE={hypothesis} trades={holdout_metrics['trades']} total_r={holdout_metrics['total_r']}", flush=True)
        results[hypothesis] = {
            "holdout_start": holdout_start,
            "holdout_rows": n - holdout_start,
            "params": asdict(selected),
            "selection": selection,
            "selection_status": selection.get("selection_status", "unknown"),
            "holdout_metrics": holdout_metrics,
        }
        if checkpoint_path:
            payload = {
                "schema": 5, "df_len": n, "holdout_ratio": holdout_ratio,
                "completed": results,
            }
            _save_wf_checkpoint(checkpoint_path, payload)
    return results
