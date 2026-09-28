from __future__ import annotations
import numpy as np

def metrics(rs):
    vals = np.asarray(list(rs), dtype=float)
    if len(vals) == 0:
        return {"trades":0,"wins":0,"losses":0,"win_rate":0.0,"total_r":0.0,"expectancy":0.0,"profit_factor":None,"max_drawdown":0.0,"max_loss_streak":0,"average_r":0.0}
    wins = vals[vals > 0]; losses = vals[vals < 0]
    equity = np.cumsum(vals); peak = np.maximum.accumulate(np.r_[0.0, equity]); dd = peak[1:] - equity
    streak = max_streak = 0
    for v in vals:
        streak = streak + 1 if v < 0 else 0
        max_streak = max(max_streak, streak)
    return {"trades":int(len(vals)),"wins":int((vals>0).sum()),"losses":int((vals<0).sum()),"win_rate":float((vals>0).mean()),"total_r":float(vals.sum()),"expectancy":float(vals.mean()),"profit_factor":(float(wins.sum()/abs(losses.sum())) if losses.size else None),"max_drawdown":float(dd.max() if len(dd) else 0.0),"max_loss_streak":int(max_streak),"average_r":float(vals.mean())}
