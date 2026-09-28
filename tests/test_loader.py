import pandas as pd
import pytest
from data.loader import validate_ohlcv

def test_validate_ohlcv_orders_and_deduplicates():
    df=pd.DataFrame([
        {"timestamp":"2026-01-01 00:05","open":1.1,"high":1.2,"low":1.0,"close":1.15},
        {"timestamp":"2026-01-01 00:00","open":1.0,"high":1.1,"low":0.9,"close":1.05},
        {"timestamp":"2026-01-01 00:00","open":1.0,"high":1.1,"low":0.9,"close":1.05}])
    out=validate_ohlcv(df); assert len(out)==2; assert out.iloc[0].timestamp < out.iloc[1].timestamp

def test_validate_ohlcv_rejects_bad_ohlc():
    df=pd.DataFrame([{"timestamp":"2026-01-01","open":1,"high":0.9,"low":0.8,"close":1.0}])
    with pytest.raises(ValueError): validate_ohlcv(df)
