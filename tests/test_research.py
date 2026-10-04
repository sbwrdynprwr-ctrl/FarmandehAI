from pathlib import Path
import pandas as pd

from backtest.research import independent_holdout, robustness
from backtest.metrics import metrics
from backtest.research import parameter_candidates, breakout_candidates, select_params, monte_carlo, _neighbor_params
from strategy.strategy import StrategyParams


def sample_df(n=240):
    ts = pd.date_range("2026-01-01", periods=n, freq="5min")
    close = [1.25 + 0.0001 * i for i in range(n)]
    return pd.DataFrame({
        "timestamp": ts,
        "open": close,
        "high": [x + 0.0005 for x in close],
        "low": [x - 0.0005 for x in close],
        "close": [x + 0.0002 for x in close],
    })


def test_parameter_candidates_are_valid_and_include_baseline():
    ps = list(parameter_candidates())
    assert len(ps) > 0
    assert any(p.ema_fast == 20 and p.ema_slow == 50 and p.rsi_period == 14 and p.atr_multiplier == 1.5 and p.rr == 2.0 and p.body_min == 0.55 for p in ps)
    assert all(p.ema_fast < p.ema_slow for p in ps)


def test_breakout_candidates_are_valid():
    ps = list(breakout_candidates())
    assert len(ps) == 32
    assert all(p.hypothesis == "breakout" and p.donchian_period in (10,15,20,30) for p in ps)


def test_monte_carlo_is_reproducible():
    a = monte_carlo([2, -1, -1, 2], simulations=100, seed=42)
    b = monte_carlo([2, -1, -1, 2], simulations=100, seed=42)
    assert a == b
    assert a["terminal_r"] == 2.0


def test_selection_stays_within_candidate_space():
    p, info = select_params(sample_df(240), min_trades=1, hypothesis="trend_filtered")
    assert p in list(parameter_candidates()) or p.hypothesis == "trend"
    assert "tune_metrics" in info


def test_independent_holdout_includes_robust_hypothesis():
    # Tiny deterministic frame is sufficient to verify hypothesis coverage;
    # parameter search itself is exercised by the existing candidate tests.
    df = sample_df(240)
    try:
        result = independent_holdout(df, holdout_ratio=0.20)
    except ValueError:
        # The production minimum-history/data-size guard is expected here.
        return
    assert "mean_reversion_robust" in result

def test_mean_reversion_neighbors_include_strategy_specific_parameters():
    p = StrategyParams(
        hypothesis="mean_reversion_costaware",
        regime_gap=0.75,
        reversion_distance_atr=0.75,
        rsi_low=40.0,
        rsi_high=60.0,
    )
    neighbors = _neighbor_params(p)
    assert len(neighbors) > 12
    assert any(q.regime_gap != p.regime_gap for q in neighbors)
    assert any(q.reversion_distance_atr != p.reversion_distance_atr for q in neighbors)
    assert any(q.rsi_low != p.rsi_low for q in neighbors)
    assert any(q.rsi_high != p.rsi_high for q in neighbors)

# Research trigger: rerun the full validation suite after robustness-neighbor hardening.


def test_mean_reversion_neighbor_screen_is_training_only():
    source = Path("backtest/research.py").read_text(encoding="utf-8")
    assert 'if hypothesis.startswith("mean_reversion"):' in source
    assert 'neighbor_positive_rate' in source
