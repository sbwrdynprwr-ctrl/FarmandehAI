"""Market-data loading with strict secret handling and historical pagination."""
from __future__ import annotations
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional
import pandas as pd
import requests
from requests import RequestException

@dataclass(frozen=True)
class DataConfig:
    symbol: str = "GBP/USD"
    interval: str = "5min"
    outputsize: int = 5000
    timezone: str = "UTC"
    timeout_seconds: int = 30

def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing: raise ValueError(f"Missing required columns: {sorted(missing)}")
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="raise")
    for c in ["open","high","low","close"]: out[c] = pd.to_numeric(out[c], errors="raise")
    if "volume" in out: out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    out = out.sort_values("timestamp", kind="stable").drop_duplicates("timestamp").reset_index(drop=True)
    if not out["timestamp"].is_monotonic_increasing: raise ValueError("Data is not chronological")
    if (out["high"] < out[["open","close"]].max(axis=1)).any() or (out["low"] > out[["open","close"]].min(axis=1)).any(): raise ValueError("Invalid OHLC relationships")
    if (out[["open","high","low","close"]] <= 0).any().any(): raise ValueError("Prices must be positive")
    return out

def load_csv(path: str) -> pd.DataFrame:
    return validate_ohlcv(pd.read_csv(path))

def _request(config: DataConfig, key: str, start: datetime | None = None, end: datetime | None = None) -> pd.DataFrame:
    params={"symbol":config.symbol,"interval":config.interval,"outputsize":config.outputsize,"timezone":config.timezone,"format":"JSON"}
    if start is not None and end is not None:
        params["start_date"]=start.strftime("%Y-%m-%dT%H:%M:%SZ"); params["end_date"]=end.strftime("%Y-%m-%dT%H:%M:%SZ"); params.pop("outputsize",None)
    try:
        response=requests.get("https://api.twelvedata.com/time_series",params=params,headers={"Authorization":f"apikey {key}"},timeout=config.timeout_seconds)
        response.raise_for_status()
        payload=response.json()
    except RequestException as exc:
        status=getattr(getattr(exc, "response", None), "status_code", None)
        detail=""
        resp=getattr(exc, "response", None)
        if resp is not None:
            try:
                body=resp.json()
                detail=str(body.get("message") or body.get("code") or "")
            except ValueError:
                detail=""
        suffix=f"; status={status}" if status is not None else ""
        if detail: suffix += f"; message={detail}"
        raise RuntimeError(f"TwelveData request failed: {exc.__class__.__name__}{suffix}") from None
    except ValueError: raise RuntimeError("TwelveData returned a non-JSON response") from None
    if payload.get("status")=="error" or "values" not in payload: raise RuntimeError(f"TwelveData error: {payload.get('message','invalid response')}")
    rows=[{"timestamp":r["datetime"],"open":r["open"],"high":r["high"],"low":r["low"],"close":r["close"],"volume":r.get("volume")} for r in payload["values"]]
    return validate_ohlcv(pd.DataFrame(rows))

def fetch_twelvedata(config: DataConfig, api_key: Optional[str]=None, days: int=30) -> pd.DataFrame:
    key=api_key or os.getenv("TWELVEDATA_API_KEY")
    if not key: raise RuntimeError("TWELVEDATA_API_KEY is missing; no valid market data is available")
    if days<1: raise ValueError("days must be >= 1")
    end=datetime.now(timezone.utc); start=end-timedelta(days=days); chunks=[]; window=timedelta(days=10); cursor=start
    while cursor<end:
        chunk_end=min(cursor+window,end); chunks.append(_request(config,key,cursor,chunk_end)); cursor=chunk_end+timedelta(minutes=5)
    if not chunks: raise RuntimeError("No market data returned")
    return validate_ohlcv(pd.concat(chunks,ignore_index=True))
