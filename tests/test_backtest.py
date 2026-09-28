import pandas as pd
from backtest.engine import run_backtest

def test_chronological_required():
    df=pd.DataFrame({"timestamp":pd.to_datetime(["2026-01-01 00:10","2026-01-01 00:05"]),"open":[1,1],"high":[1,1],"low":[1,1],"close":[1,1]})
    try: run_backtest(df)
    except ValueError: return
    assert False
