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

def run_backtest(df: pd.DataFrame, params=StrategyParams(), initial_equity=10000.0,
                 spread=0.0, start_index=None, end_index=None):
    """Run the original bar-by-bar model with NumPy arrays for the hot exit loop.

    Trading semantics are unchanged: signal on bar i, enter at i+1 open,
    SL/TP are checked from the entry bar onward, and when both are hit on
    one candle SL is resolved first.
    """
    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("Backtest requires chronological data")
    x = indicators(df, params)
    n = len(x)
    start = 1 if start_index is None else max(1, int(start_index))
    stop = n - 2 if end_index is None else min(n - 2, int(end_index) - 1)
    if start > stop:
        return []

    # Extract only columns used by the execution loop once. Avoid repeated
    # DataFrame iloc/scalar access for every candle in every candidate.
    opens = x["open"].to_numpy(dtype=float)
    highs = x["high"].to_numpy(dtype=float)
    lows = x["low"].to_numpy(dtype=float)
    atrs = x["atr"].to_numpy(dtype=float)
    timestamps = x["timestamp"].to_numpy()
    trades = []
    i = start

    while i <= stop:
        side = signal_at(x, i, params)
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
        else:
            entry -= spread / 2
            sl = entry + risk
            tp = entry - risk * params.rr

        exit_price = None
        reason = None
        j = i + 1
        if side == "LONG":
            while j <= stop:
                lo = lows[j]
                hi = highs[j]
                hit_sl = lo <= sl
                hit_tp = hi >= tp
                if hit_sl and hit_tp:
                    exit_price = sl
                    reason = "SL_AND_TP_SAME_BAR_SL_FIRST"
                    break
                if hit_sl:
                    exit_price = sl
                    reason = "STOP_LOSS"
                    break
                if hit_tp:
                    exit_price = tp
                    reason = "TAKE_PROFIT"
                    break
                j += 1
        else:
            while j <= stop:
                lo = lows[j]
                hi = highs[j]
                hit_sl = hi >= sl
                hit_tp = lo <= tp
                if hit_sl and hit_tp:
                    exit_price = sl
                    reason = "SL_AND_TP_SAME_BAR_SL_FIRST"
                    break
                if hit_sl:
                    exit_price = sl
                    reason = "STOP_LOSS"
                    break
                if hit_tp:
                    exit_price = tp
                    reason = "TAKE_PROFIT"
                    break
                j += 1

        if exit_price is None:
            break

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


def run_backtest_window(df: pd.DataFrame, params=StrategyParams(),
                        start_index: int = 0, end_index=None,
                        spread: float = 0.0):
    """Backtest an OOS window with prior candles available for indicator warm-up.

    Signals may use the last completed candle before start_index so an entry at
    the first OOS candle is evaluated correctly. No candle after end_index is
    used for signals or exits.
    """
    if end_index is None:
        end_index = len(df)
    start_index = int(start_index)
    end_index = int(end_index)
    if not 0 <= start_index < end_index <= len(df):
        raise ValueError("invalid backtest window")
    signal_start = max(1, start_index - 1)
    return run_backtest(
        df,
        params=params,
        spread=spread,
        start_index=signal_start,
        end_index=end_index,
    )
