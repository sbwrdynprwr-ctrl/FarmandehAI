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
    monkeypatch.setattr(engine, "signal_at", lambda x, i, params: "LONG" if i == 1 else None)
    trades = run_backtest(df, params=StrategyParams(atr_multiplier=1.0))
    assert len(trades) == 1
    assert trades[0].reason == "SL_AND_TP_SAME_BAR_SL_FIRST"
    assert trades[0].exit == trades[0].stop_loss
    assert trades[0].r == pytest.approx(-1.0)


