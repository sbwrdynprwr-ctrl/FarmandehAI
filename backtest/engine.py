from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from strategy.strategy import StrategyParams, indicators, signal_at

@dataclass
class Trade:
    timestamp: object; side: str; entry: float; stop_loss: float; take_profit: float; exit: float; result: str; r: float; reason: str

def run_backtest(df: pd.DataFrame, params=StrategyParams(), initial_equity=10000.0, spread=0.0):
    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("Backtest requires chronological data")
    x = indicators(df, params)
    trades=[]; i=1
    while i < len(x)-1:
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
        while j < len(x):
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
        r = (exit_price-entry)/risk if side=="LONG" else (entry-exit_price)/risk
        trades.append(Trade(x.iloc[i].timestamp, side, entry, sl, tp, exit_price, "WIN" if r>0 else "LOSS", float(r), reason))
        i=j+1
    return trades
