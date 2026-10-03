import pandas as pd
import pytest
import backtest.engine as engine
from backtest.engine import run_backtest
from strategy.strategy import StrategyParams

def test_chronological_required():
    df = pd.DataFrame({"timestamp": pd.to_datetime(["2026-01-01 00:10", "2026-01-01 00:05"]), "open": [1,1], "high": [1,1], "low": [1,1], "close": [1,1]})
    try:
        run_backtest(df)
    except ValueError:
        return
    assert False

def test_same_bar_sl_and_tp_is_reproducibly_sl_first(monkeypatch):
    df = pd.DataFrame({"timestamp": pd.to_datetime(["2026-01-01 00:00","2026-01-01 00:05","2026-01-01 00:10"]), "open": [1.0,1.0,1.0], "high": [1.0,1.0,1.2], "low": [1.0,1.0,0.8], "close": [1.0,1.0,1.0]})
    fake = df.copy()
    fake["atr"] = 0.1
    monkeypatch.setattr(engine, "indicators", lambda data, params: fake)
    monkeypatch.setattr(engine, "_signal_array", lambda x, params: __import__("numpy").array(["", "LONG", ""], dtype=object))
    trades = run_backtest(df, p=StrategyParams(atr_multiplier=1.0))
    assert len(trades) == 1
    assert trades[0].reason == "SL_AND_TP_SAME_BAR_SL_FIRST"
    assert trades[0].exit == trades[0].stop_loss
    assert trades[0].r == pytest.approx(-1.0)




def test_spread_is_round_trip_cost(monkeypatch):
    df = pd.DataFrame({"timestamp": pd.to_datetime(["2026-01-01 00:00","2026-01-01 00:05","2026-01-01 00:10"]), "open": [1.0,1.0,1.0], "high": [1.0,1.0,1.3], "low": [1.0,1.0,1.1], "close": [1.0,1.0,1.0]})
    fake = df.copy()
    fake["atr"] = 0.1
    monkeypatch.setattr(engine, "indicators", lambda data, params: fake)
    monkeypatch.setattr(engine, "_signal_array", lambda x, params: __import__("numpy").array(["", "LONG", ""], dtype=object))
    no_spread = run_backtest(df, p=StrategyParams(atr_multiplier=1.0), spread=0.0)
    with_spread = run_backtest(df, p=StrategyParams(atr_multiplier=1.0), spread=0.02)
    assert no_spread[0].r == pytest.approx(2.0)
    assert with_spread[0].r < no_spread[0].r


def test_oos_window_uses_prior_candles_for_warmup(monkeypatch):
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=6, freq="5min"),
        "open": [1.0]*6, "high": [1.2]*6, "low": [0.8]*6, "close": [1.0]*6,
    })
    fake = df.copy()
    fake["atr"] = 0.1
    monkeypatch.setattr(engine, "indicators", lambda data, params: fake)
    monkeypatch.setattr(engine, "_signal_array", lambda x, params: __import__("numpy").array(["", "", "LONG", "", "", ""], dtype=object))
    trades = engine.run_backtest_window(df, StrategyParams(atr_multiplier=1.0), start_index=3, end_index=6)
    assert len(trades) == 1
    assert trades[0].timestamp == df.iloc[2].timestamp

def test_costaware_mean_reversion_requires_range_and_distance(monkeypatch):
    import numpy as np
    n = 40
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="5min"),
        "open": [1.0] * n, "high": [1.2] * n, "low": [0.8] * n, "close": [1.0] * n,
    })
    fake = df.copy()
    fake["atr"] = 0.1
    fake["ema_fast"] = 1.0
    fake["ema_slow"] = 1.0
    fake["rsi"] = 35.0
    fake["body_ratio"] = 0.8
    fake["macd"] = 0.0
    fake["macd_signal"] = 0.0
    fake["bb_mid"] = 1.1
    fake["bb_upper"] = 1.2
    fake["bb_lower"] = 0.9
    fake["trend_gap_atr"] = 0.3
    fake.loc[35, "open"] = 0.85
    fake.loc[35, "close"] = 0.9
    monkeypatch.setattr(engine, "indicators", lambda data, params: fake)
    params = StrategyParams(
        hypothesis="mean_reversion_costaware",
        rsi_low=40.0,
        rsi_high=60.0,
        regime_gap=0.5,
        reversion_distance_atr=1.0,
    )
    signals = engine._signal_array(fake, params)
    assert signals[35] == "LONG"
    fake.loc[35, "trend_gap_atr"] = 0.8
    signals = engine._signal_array(fake, params)
    assert signals[35] == ""

