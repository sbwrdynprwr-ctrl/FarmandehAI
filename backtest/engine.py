from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from strategy.strategy import StrategyParams, indicators

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

def _signal_array(x, p):
    n = len(x)
    out = np.full(n, "", dtype=object)
    body = x["body_ratio"].to_numpy(float)
    atr = x["atr"].to_numpy(float)
    ef = x["ema_fast"].to_numpy(float)
    es = x["ema_slow"].to_numpy(float)
    rsi = x["rsi"].to_numpy(float)
    macd = x["macd"].to_numpy(float)
    sig = x["macd_signal"].to_numpy(float)
    op = x["open"].to_numpy(float)
    cl = x["close"].to_numpy(float)
    valid = np.isfinite(ef)&np.isfinite(es)&np.isfinite(rsi)&np.isfinite(atr)&np.isfinite(body)&np.isfinite(macd)&np.isfinite(sig)
    valid &= body >= p.body_min
    if p.hypothesis in ("mean_reversion", "mean_reversion_v2"):
        up=x["bb_upper"].to_numpy(float); lo=x["bb_lower"].to_numpy(float)
        mid=x["bb_mid"].to_numpy(float)
        valid &= np.isfinite(mid)
        if p.hypothesis == "mean_reversion_v2":
            long_mask=valid&(cl<mid)&(rsi<=p.rsi_low)&(cl>op)
            short_mask=valid&(cl>mid)&(rsi>=p.rsi_high)&(cl<op)
            mh=max(p.rsi_period,p.atr_period,30)
            long_mask[:mh]=False; short_mask[:mh]=False
            out[long_mask]="LONG"; out[short_mask]="SHORT"; return out
        valid &= np.isfinite(up)&np.isfinite(lo)
        long_mask=valid&(cl<=lo)&(rsi<=35)&(cl>op)
        short_mask=valid&(cl>=up)&(rsi>=65)&(cl<op)
        mh=max(p.rsi_period,p.atr_period,30)
    elif p.hypothesis == "breakout":
        up=x["donchian_upper"].to_numpy(float); lo=x["donchian_lower"].to_numpy(float)
        valid &= np.isfinite(up)&np.isfinite(lo)
        long_mask=valid&(cl>up)&(cl>op)
        short_mask=valid&(cl<lo)&(cl<op)
        mh=max(p.atr_period,p.donchian_period,30)
    elif p.hypothesis == "pullback":
        long_mask=valid&(ef>es)&(rsi>=40)&(rsi<=50)&(macd>=sig)&(cl>op)
        short_mask=valid&(ef<es)&(rsi>=50)&(rsi<=60)&(macd<=sig)&(cl<op)
        mh=max(p.ema_slow,p.rsi_period,p.atr_period,30)
    elif p.hypothesis in ("trend_filtered", "trend_regime"):
        slope=x["ema_slope"].to_numpy(float)
        gap=x["trend_gap_atr"].to_numpy(float)
        valid &= np.isfinite(slope)&np.isfinite(gap)
        s=p.trend_strength
        if p.hypothesis == "trend_regime":
            persistence=x["trend_persistence"].to_numpy(float)
            valid &= np.isfinite(persistence)
            long_mask=valid&(ef>es)&(slope>s*atr)&(gap>=p.regime_gap)&(persistence>=0.75)&(rsi>=p.rsi_low)&(rsi<=p.rsi_high)&(macd>sig)&(cl>op)&(cl>ef)
            short_mask=valid&(ef<es)&(slope<-s*atr)&(gap>=p.regime_gap)&(persistence<=0.25)&(rsi>=(100-p.rsi_high))&(rsi<=(100-p.rsi_low))&(macd<sig)&(cl<op)&(cl<ef)
            mh=max(p.ema_slow,p.rsi_period,p.atr_period,p.slope_bars+4,35)
        else:
            long_mask=valid&(ef>es)&(slope>s*atr)&(gap>=s)&(rsi>=p.rsi_low)&(rsi<=p.rsi_high)&(macd>sig)&(cl>op)&(cl>ef)
            short_mask=valid&(ef<es)&(slope<-s*atr)&(gap>=s)&(rsi>=(100-p.rsi_high))&(rsi<=(100-p.rsi_low))&(macd<sig)&(cl<op)&(cl<ef)
            mh=max(p.ema_slow,p.rsi_period,p.atr_period,35)
    else:
        long_mask=valid&(ef>es)&(rsi>=50)&(macd>sig)&(cl>op)
        short_mask=valid&(ef<es)&(rsi<=50)&(macd<sig)&(cl<op)
        mh=max(p.ema_slow,p.rsi_period,p.atr_period,30)
    long_mask[:mh]=False
    short_mask[:mh]=False
    out[long_mask]="LONG"; out[short_mask]="SHORT"
    return out

class _RangeFirstHit:
    def __init__(self, values):
        a=np.asarray(values,float); self.n=len(a); size=1
        while size<self.n: size<<=1
        self.size=size; self.lo=np.full(2*size,np.inf); self.hi=np.full(2*size,-np.inf)
        self.lo[size:size+self.n]=a; self.hi[size:size+self.n]=a
        for k in range(size-1,0,-1):
            self.lo[k]=min(self.lo[2*k],self.lo[2*k+1]); self.hi[k]=max(self.hi[2*k],self.hi[2*k+1])
    def first_le(self,l,r,t): return self._first(l,r,t,True)
    def first_ge(self,l,r,t): return self._first(l,r,t,False)
    def _first(self,l,r,t,le):
        if l>r or not self.n: return None
        def visit(node,nl,nr):
            if nr<l or r<nl: return None
            b=self.lo[node] if le else self.hi[node]
            if (b>t) if le else (b<t): return None
            if nl==nr: return nl if nl<self.n else None
            m=(nl+nr)//2
            h=visit(node*2,nl,m)
            return h if h is not None else visit(node*2+1,m+1,nr)
        return visit(1,0,self.size-1)

def _run_backtest_indicators(x,p=StrategyParams(),initial_equity=10000.0,spread=0.0,start_index=None,end_index=None):
    if not x["timestamp"].is_monotonic_increasing: raise ValueError("Backtest requires chronological data")
    n=len(x); start=1 if start_index is None else max(1,int(start_index)); last_signal=n-2 if end_index is None else min(n-2,int(end_index)-2); stop=n-1 if end_index is None else min(n-1,int(end_index)-1)
    if start>last_signal: return []
    signals=_signal_array(x,p); op=x["open"].to_numpy(float); hi=x["high"].to_numpy(float); lo=x["low"].to_numpy(float); atr=x["atr"].to_numpy(float); ts=x["timestamp"].to_numpy()
    lt=_RangeFirstHit(lo); ht=_RangeFirstHit(hi); trades=[]; i=start
    while i<=last_signal:
        side=signals[i]
        if not side: i+=1; continue
        entry=float(op[i+1]); risk=p.atr_multiplier*float(atr[i])
        if not np.isfinite(risk) or risk<=0: i+=1; continue
        if side=="LONG":
            entry+=spread/2; sl=entry-risk; tp=entry+risk*p.rr; si=lt.first_le(i+1,stop,sl); ti=ht.first_ge(i+1,stop,tp)
        else:
            entry-=spread/2; sl=entry+risk; tp=entry-risk*p.rr; si=ht.first_ge(i+1,stop,sl); ti=lt.first_le(i+1,stop,tp)
        if si is None and ti is None: break
        if si is None or (ti is not None and ti<si): j=ti; ep=tp; reason="TAKE_PROFIT"
        elif ti is None or si<ti: j=si; ep=sl; reason="STOP_LOSS"
        else: j=si; ep=sl; reason="SL_AND_TP_SAME_BAR_SL_FIRST"
        ex=ep-spread/2 if side=="LONG" else ep+spread/2
        rr=(ex-entry)/risk if side=="LONG" else (entry-ex)/risk
        trades.append(Trade(ts[i],side,entry,sl,tp,ep,"WIN" if rr>0 else "LOSS",float(rr),reason))
        i=j+1
    return trades

def run_backtest(df,p=StrategyParams(),initial_equity=10000.0,spread=0.0,start_index=None,end_index=None):
    if not df["timestamp"].is_monotonic_increasing: raise ValueError("Backtest requires chronological data")
    return _run_backtest_indicators(indicators(df,p),p,initial_equity,spread,start_index,end_index)

def run_backtest_window(df,p=StrategyParams(),start_index=0,end_index=None,spread=0.0):
    if end_index is None: end_index=len(df)
    start_index=int(start_index); end_index=int(end_index)
    if not 0<=start_index<end_index<=len(df): raise ValueError("invalid backtest window")
    return run_backtest(df,p,spread=spread,start_index=max(1,start_index-1),end_index=end_index)
