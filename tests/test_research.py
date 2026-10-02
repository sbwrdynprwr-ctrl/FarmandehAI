import pandas as pd

from backtest.metrics import metrics
from backtest.research import parameter_candidates, breakout_candidates, select_params, monte_carlo


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
    assert p in list(parameter_candidates())
    assert "tune_metrics" in info
