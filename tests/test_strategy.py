import pandas as pd
from strategy.strategy import indicators

def test_indicators_do_not_change_prior_values_when_future_row_added():
    base=pd.DataFrame({"timestamp":pd.date_range("2026-01-01",periods=80,freq="5min"),"open":range(80),"high":[x+1 for x in range(80)],"low":range(80),"close":[x+.5 for x in range(80)]})
    p=__import__('strategy.strategy',fromlist=['StrategyParams']).StrategyParams()
    a=indicators(base.iloc[:70].copy(),p); b=indicators(base,p)
    assert a.iloc[-1][["ema_fast","ema_slow","rsi","atr"]].equals(b.iloc[69][["ema_fast","ema_slow","rsi","atr"]])
