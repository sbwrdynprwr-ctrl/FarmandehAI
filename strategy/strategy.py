from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class StrategyParams:
    ema_fast: int = 20
    ema_slow: int = 50
    rsi_period: int = 14
    atr_period: int = 14
    atr_multiplier: float = 1.5
    rr: float = 2.0
    body_min: float = 0.55

def indicators(df: pd.DataFrame, p: StrategyParams) -> pd.DataFrame:
    x = df.copy()
    close, high, low = x["close"], x["high"], x["low"]
    x["ema_fast"] = close.ewm(span=p.ema_fast, adjust=False).mean()
    x["ema_slow"] = close.ewm(span=p.ema_slow, adjust=False).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(p.rsi_period).mean()
    loss = (-delta.clip(upper=0)).rolling(p.rsi_period).mean()
    rs = gain / loss.replace(0, np.nan)
    x["rsi"] = 100 - (100 / (1 + rs))
    prev_close = close.shift(1)
    tr = pd.concat([(high-low), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    x["atr"] = tr.rolling(p.atr_period).mean()
    x["body_ratio"] = (close-x["open"]).abs() / (high-low).replace(0, np.nan)
    macd_fast = close.ewm(span=12, adjust=False).mean()
    macd_slow = close.ewm(span=26, adjust=False).mean()
    x["macd"] = macd_fast - macd_slow
    x["macd_signal"] = x["macd"].ewm(span=9, adjust=False).mean()
    return x

def signal_at(x: pd.DataFrame, i: int, p: StrategyParams):
    if i < max(p.ema_slow, p.rsi_period, p.atr_period, 30):
        return None
    r = x.iloc[i]
    if not np.isfinite(r[["ema_fast","ema_slow","rsi","atr","macd","macd_signal","body_ratio"]].to_numpy(dtype=float)).all():
        return None
    if r.body_ratio < p.body_min:
        return None
    if r.ema_fast > r.ema_slow and r.rsi >= 50 and r.macd > r.macd_signal and r.close > r.open:
        return "LONG"
    if r.ema_fast < r.ema_slow and r.rsi <= 50 and r.macd < r.macd_signal and r.close < r.open:
        return "SHORT"
    return None
