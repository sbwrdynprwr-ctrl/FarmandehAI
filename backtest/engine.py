from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from strategy.strategy import StrategyParams, indicators, signal_at

@dataclass
class Trade:
    timestamp: object
    side: str
    entry: float
    stop_loss: float
    take_profit: float
    exit: float
    result: str
    r: float
    reason: str

def _signal_array(x: pd.DataFrame, p: StrategyParams) -> np.ndarray:
    """Vectorized equivalent of signal_at for the full indicator frame."""
    n = len(x)
    out = np.full(n, "", dtype=object)
    body = x["body_ratio"].to_numpy(dtype=float)
    atr = x["atr"].to_numpy(dtype=float)

    if p.hypothesis == "mean_reversion":
        rsi = x["rsi"].to_numpy(dtype=float)
        upper = x["bb_upper"].to_numpy(dtype=float)
        lower = x["bb_lower"].to_numpy(dtype=float)
        op = x["open"].to_numpy(dtype=float)
        cl = x["close"].to_numpy(dtype=float)
        valid = np.isfinite(rsi) & np.isfinite(atr) & np.isfinite(body) & np.isfinite(upper) & np.isfinite(lower)
        valid &= body >= p.body_min
        long_mask = valid & (cl <= lower) & (rsi <= 35) & (cl > op)
        short_mask = valid & (cl >= upper) & (rsi >= 65) & (cl < op)
        min_history = max(p.rsi_period, p.atr_period, 30)
    elif p.hypothesis == "breakout":
        upper = x["donchian_upper"].to_numpy(dtype=float)
        lower = x["donchian_lower"].to_numpy(dtype=float)
        op = x["open"].to_numpy(dtype=float)
        cl = x["close"].to_numpy(dtype=float)
        valid = np.isfinite(atr) & np.isfinite(body) & np.isfinite(upper) & np.isfinite(lower)
        valid &= body >= p.body_min
        long_mask = valid & (cl > upper) & (cl > op)
        short_mask = valid & (cl < lower) & (cl < op)
        min_history = max(p.atr_period, p.donchian_period, 30)
    elif p.hypothesis == "pullback":
        ef = x["ema_fast"].to_numpy(dtype=float)
        es = x["ema_slow"].to_numpy(dtype=float)
        rsi = x["rsi"].to_numpy(dtype=float)
        macd = x["macd"].to_numpy(dtype=float)
        sig = x["macd_signal"].to_numpy(dtype=float)
        op = x["open"].to_numpy(dtype=float)
        cl = x["close"].to_numpy(dtype=float)
        valid = np.isfinite(ef) & np.isfinite(es) & np.isfinite(rsi) & np.isfinite(atr) & np.isfinite(body) & np.isfinite(macd) & np.isfinite(sig)
        valid &= body >= p.body_min
        long_mask = valid & (ef > es) & (rsi >= 40) & (rsi <= 50) & (macd >= sig) & (cl > op)
        short_mask = valid & (ef < es) & (rsi >= 50) & (rsi <= 60) & (macd <= sig) & (cl < op)
        min_history = max(p.ema_slow, p.rsi_period, p.atr_period, 30)
    else:
        ef = x["ema_fast"].to_numpy(dtype=float)
        es = x["ema_slow"].to_numpy(dtype=float)
        rsi = x["rsi"].to_numpy(dtype=float)
        macd = x["macd"].to_numpy(dtype=float)
        sig = x["macd_signal"].to_numpy(dtype=float)
        op = x["open"].to_numpy(dtype=float)
        cl = x["close"].to_numpy(dtype=float)
        valid = np.isfinite(ef) & np.isfinite(es) & np.isfinite(rsi) & np.isfinite(atr) & np.isfinite(macd) & np.isfinite(sig) & np.isfinite(body)
        valid &= body >= p.body_min
        long_mask = valid & (ef > es) & (rsi >= 50) & (macd > sig) & (cl > op)
        short_mask = valid & (ef < es) & (rsi <= 50) & (macd < sig) & (cl < op)
        min_history = max(p.ema_slow, p.rsi_period, p.atr_period, 30)

    long_mask[:min_history] = False
    short_mask[:min_history] = False
    out[long_mask] = "LONG"
    out[short_mask] = "SHORT"
    return out


class _RangeFirstHit:
    """Static segment tree for leftmost threshold hits in a price series."""
    def __init__(self, values):
        a = np.asarray(values, dtype=float)
        self.n = len(a)
        size = 1
        while size < self.n:
            size <<= 1
        self.size = size
        self.lo = np.full(2 * size, np.inf, dtype=float)
        self.hi = np.full(2 * size, -np.inf, dtype=float)
        self.lo[size:size + self.n] = a
        self.hi[size:size + self.n] = a
        for k in range(size - 1, 0, -1):
            self.lo[k] = min(self.lo[2*k], self.lo[2*k+1])
            self.hi[k] = max(self.hi[2*k], self.hi[2*k+1])

    def first_le(self, left, right, threshold):
        return self._first(left, right, threshold, le=True)

    def first_ge(self, left, right, threshold):
        return self._first(left, right, threshold, le=False)

    def _first(self, left, right, threshold, le):
        if left > right or self.n == 0:
            return None
        def visit(node, nl, nr):
            if nr < left or right < nl:
                return None
            bound = self.lo[node] if le else self.hi[node]
            if (bound > threshold) if le else (bound < threshold):
                return None
            if nl == nr:
                return nl if nl < self.n else None
            mid = (nl + nr) // 2
            hit = visit(node * 2, nl, mid)
            if hit is not None:
                return hit
            return visit(node * 2 + 1, mid + 1, nr)
        return visit(1, 0, self.size - 1)

def _run_backtest_indicators(x: pd.DataFrame, params=StrategyParams(),
                             initial_equity=10000.0, spread=0.0,
                             start_index=None, end_index=None):
    """Fast equivalent of the original bar-by-bar backtest.

    Signal/entry/SL/TP semantics are unchanged. Same-bar SL+TP resolves to SL.
    """
    if not x["timestamp"].is_monotonic_increasing:
        raise ValueError("Backtest requires chronological data")
    n = len(x)
    start = 1 if start_index is None else max(1, int(start_index))
    stop = n - 2 if end_index is None else min(n - 2, int(end_index) - 1)
    if start > stop:
        return []

    signals = _signal_array(x, params)
    opens = x["open"].to_numpy(dtype=float)
    highs = x["high"].to_numpy(dtype=float)
    lows = x["low"].to_numpy(dtype=float)
    atrs = x["atr"].to_numpy(dtype=float)
    timestamps = x["timestamp"].to_numpy()
    low_tree = _RangeFirstHit(lows)
    high_tree = _RangeFirstHit(highs)

    trades = []
    i = start
    while i <= stop:
        side = signals[i]
        if not side:
            i += 1
            continue

        entry = float(opens[i + 1])
        atr = float(atrs[i])
        risk = params.atr_multiplier * atr
        if not np.isfinite(risk) or risk <= 0:
            i += 1
            continue

        if side == "LONG":
            entry += spread / 2
            sl = entry - risk
            tp = entry + risk * params.rr
            sl_idx = low_tree.first_le(i + 1, stop, sl)
            tp_idx = high_tree.first_ge(i + 1, stop, tp)
        else:
            entry -= spread / 2
            sl = entry + risk
            tp = entry - risk * params.rr
            sl_idx = high_tree.first_ge(i + 1, stop, sl)
            tp_idx = low_tree.first_le(i + 1, stop, tp)
        if sl_idx is None and tp_idx is None:
            break

        if sl_idx is None or (tp_idx is not None and tp_idx < sl_idx):
            j = tp_idx
            exit_price = tp
            reason = "TAKE_PROFIT"
        elif tp_idx is None or sl_idx < tp_idx:
            j = sl_idx
            exit_price = sl
            reason = "STOP_LOSS"
        else:
            j = sl_idx
            exit_price = sl
            reason = "SL_AND_TP_SAME_BAR_SL_FIRST"

        executed_exit = (
            exit_price - spread / 2 if side == "LONG"
            else exit_price + spread / 2
        )
        r = (
            (executed_exit - entry) / risk if side == "LONG"
            else (entry - executed_exit) / risk
        )
        trades.append(Trade(
            timestamps[i], side, entry, sl, tp, exit_price,
            "WIN" if r > 0 else "LOSS", float(r), reason
        ))
        i = j + 1

    return trades

def run_backtest(df: pd.DataFrame, params=StrategyParams(), initial_equity=10000.0,
                 spread=0.0, start_index=None, end_index=None):
    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("Backtest requires chronological data")
    x = indicators(df, params)
    return _run_backtest_indicators(x, params, initial_equity, spread, start_index, end_index)


def run_backtest_window(df: pd.DataFrame, params=StrategyParams(),
                        start_index: int = 0, end_index=None,
                        spread: float = 0.0):
    """Backtest an OOS window with prior candles available for indicator warm-up."""
    if end_index is None:
        end_index = len(df)
    start_index = int(start_index)
    end_index = int(end_index)
    if not 0 <= start_index < end_index <= len(df):
        raise ValueError("invalid backtest window")
    signal_start = max(1, start_index - 1)
    return run_backtest(df, params=params, spread=spread,
                        start_index=signal_start, end_index=end_index)
