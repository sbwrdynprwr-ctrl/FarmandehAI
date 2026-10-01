from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from strategy.strategy import StrategyParams, indicators, signal_at

@dataclass
class Trade:
    timestamp: object; side: str; entry: float; stop_loss: float; take_profit: float; exit: float; result: str; r: float; reason: str

def run_backtest(df: pd.DataFrame, params=StrategyParams(), initial_equity=10000.0, spread=0.0, start_index=None, end_index=None):
    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("Backtest requires chronological data")
    x = indicators(df, params)
    start = 1 if start_index is None else max(1, int(start_index))
    # A signal at i enters on candle i+1, so i must be at most len(x)-2.
    stop = len(x) - 2 if end_index is None else min(len(x) - 2, int(end_index) - 1)
    if start > stop:
        return []
    trades=[]; i=start
    while i <= stop:
        side = signal_at(x, i, params)
        if not side:
            i += 1; continue
        entry = float(x.iloc[i+1].open)
        atr = float(x.iloc[i].atr)
        risk = params.atr_multiplier * atr
        if risk <= 0:
            i += 1; continue
        if side == "LONG":
            entry += spread/2; sl=entry-risk; tp=entry+risk*params.rr
        else:
            entry -= spread/2; sl=entry+risk; tp=entry-risk*params.rr
        exit_price=None; reason=None; j=i+1
        while j <= stop:
            row=x.iloc[j]
            if side == "LONG":
                hit_sl=row.low <= sl; hit_tp=row.high >= tp
            else:
                hit_sl=row.high >= sl; hit_tp=row.low <= tp
            if hit_sl and hit_tp:
                exit_price=sl; reason="SL_AND_TP_SAME_BAR_SL_FIRST"; break
            if hit_sl:
                exit_price=sl; reason="STOP_LOSS"; break
            if hit_tp:
                exit_price=tp; reason="TAKE_PROFIT"; break
            j += 1
        if exit_price is None: break
        executed_exit = exit_price - spread/2 if side == "LONG" else exit_price + spread/2
        r = (executed_exit-entry)/risk if side=="LONG" else (entry-executed_exit)/risk
        trades.append(Trade(x.iloc[i].timestamp, side, entry, sl, tp, exit_price, "WIN" if r>0 else "LOSS", float(r), reason))
        i=j+1
    return trades


def run_backtest_window(df: pd.DataFrame, params=StrategyParams(), start_index: int = 0, end_index=None, spread: float = 0.0):
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
