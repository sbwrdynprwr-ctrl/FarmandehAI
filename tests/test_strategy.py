import pandas as pd
from strategy.strategy import indicators, StrategyParams, signal_at

def test_indicators_do_not_change_prior_values_when_future_row_added():
    base=pd.DataFrame({"timestamp":pd.date_range("2026-01-01",periods=80,freq="5min"),"open":range(80),"high":[x+1 for x in range(80)],"low":range(80),"close":[x+.5 for x in range(80)]})
    p=StrategyParams()
    a=indicators(base.iloc[:70].copy(),p); b=indicators(base,p)
    assert a.iloc[-1][["ema_fast","ema_slow","rsi","atr"]].equals(b.iloc[69][["ema_fast","ema_slow","rsi","atr"]])

def test_breakout_bands_are_shifted_and_signal_uses_prior_range():
    ts=pd.date_range("2026-01-01",periods=40,freq="5min")
    close=[1.0]*39+[1.01]
    df=pd.DataFrame({
        "timestamp":ts, "open":[1.0]*40,
        "high":[1.001]*39+[1.011],
        "low":[0.999]*40, "close":close,
    })
    p=StrategyParams(hypothesis="breakout", donchian_period=20, body_min=0.1)
    x=indicators(df,p)
    assert x.iloc[39].donchian_upper == x.iloc[19:39].high.max()
    assert signal_at(x,39,p) == "LONG"
