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
    hypothesis: str = "trend"
    donchian_period: int = 20

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
    bb_mid = close.rolling(p.rsi_period).mean()
    bb_std = close.rolling(p.rsi_period).std(ddof=0)
    x["bb_mid"] = bb_mid
    x["bb_upper"] = bb_mid + 2.0 * bb_std
    x["bb_lower"] = bb_mid - 2.0 * bb_std
    # Shifted Donchian bands ensure the signal only uses completed prior candles.
    x["donchian_upper"] = high.rolling(p.donchian_period).max().shift(1)
    x["donchian_lower"] = low.rolling(p.donchian_period).min().shift(1)
    return x

def signal_at(x: pd.DataFrame, i: int, p: StrategyParams):
    if i < max(p.ema_slow, p.rsi_period, p.atr_period, p.donchian_period, 30):
        return None
    r = x.iloc[i]
    if p.hypothesis == "mean_reversion":
        needed = ["rsi", "atr", "body_ratio", "bb_upper", "bb_lower"]
        if not np.isfinite(r[needed].to_numpy(dtype=float)).all() or r.body_ratio < p.body_min:
            return None
        if r.close <= r.bb_lower and r.rsi <= 35 and r.close > r.open:
            return "LONG"
        if r.close >= r.bb_upper and r.rsi >= 65 and r.close < r.open:
            return "SHORT"
        return None
    if p.hypothesis == "breakout":
        needed = ["atr", "body_ratio", "donchian_upper", "donchian_lower"]
        if not np.isfinite(r[needed].to_numpy(dtype=float)).all() or r.body_ratio < p.body_min:
            return None
        if r.close > r.donchian_upper and r.close > r.open:
            return "LONG"
        if r.close < r.donchian_lower and r.close < r.open:
            return "SHORT"
        return None
    needed = ["ema_fast","ema_slow","rsi","atr","macd","macd_signal","body_ratio"]
    if not np.isfinite(r[needed].to_numpy(dtype=float)).all() or r.body_ratio < p.body_min:
        return None
    if r.ema_fast > r.ema_slow and r.rsi >= 50 and r.macd > r.macd_signal and r.close > r.open:
        return "LONG"
    if r.ema_fast < r.ema_slow and r.rsi <= 50 and r.macd < r.macd_signal and r.close < r.open:
        return "SHORT"
    return None
