"""Market-data loading with strict validation and optional public historical source."""
from __future__ import annotations
import os
import time
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
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="raise")
    for c in ["open", "high", "low", "close"]:
        out[c] = pd.to_numeric(out[c], errors="raise")
    if "volume" in out:
        out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    out = (
        out.sort_values("timestamp", kind="stable")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )
    if not out["timestamp"].is_monotonic_increasing:
        raise ValueError("Data is not chronological")
    if (
        (out["high"] < out[["open", "close"]].max(axis=1)).any()
        or (out["low"] > out[["open", "close"]].min(axis=1)).any()
    ):
        raise ValueError("Invalid OHLC relationships")
    if (out[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Prices must be positive")
    return out

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns and "datetime" in df.columns:
        df = df.rename(columns={"datetime": "timestamp"})
    return validate_ohlcv(df)

def load_remote_csv(url: str, timeout: int = 60) -> pd.DataFrame:
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
    except RequestException as exc:
        raise RuntimeError(
            f"Historical CSV download failed: {exc.__class__.__name__}"
        ) from None
    from io import StringIO
    df = pd.read_csv(StringIO(response.text))
    if "timestamp" not in df.columns and "datetime" in df.columns:
        df = df.rename(columns={"datetime": "timestamp"})
    return validate_ohlcv(df)

def load_research_data(config: DataConfig, days: int = 30) -> pd.DataFrame:
    """Load validated CSV, remote CSV, or TwelveData in that order."""
    csv_path = os.getenv("RESEARCH_DATA_CSV")
    if csv_path:
        if not os.path.isfile(csv_path):
            raise RuntimeError(f"RESEARCH_DATA_CSV does not exist: {csv_path}")
        df = load_csv(csv_path)
        if df.empty:
            raise RuntimeError("RESEARCH_DATA_CSV contains no rows")
        print(f"FARMANDEHAI_DATA_SOURCE=csv:{csv_path}", flush=True)
        return df

    remote_url = os.getenv("RESEARCH_DATA_URL")
    if remote_url:
        df = load_remote_csv(remote_url)
        if df.empty:
            raise RuntimeError("RESEARCH_DATA_URL returned no rows")
        if days > 0:
            cutoff = df["timestamp"].max() - pd.Timedelta(days=days)
            df = df[df["timestamp"] >= cutoff].reset_index(drop=True)
        print(f"FARMANDEHAI_DATA_SOURCE=remote_csv:{remote_url}", flush=True)
        print(f"FARMANDEHAI_DATA_COVERAGE={df['timestamp'].min()}..{df['timestamp'].max()}", flush=True)
        return df

    print("FARMANDEHAI_DATA_SOURCE=twelvedata", flush=True)
    return fetch_twelvedata(config, days=days)

def _request(config: DataConfig, key: str, start: datetime | None = None, end: datetime | None = None) -> pd.DataFrame:
    params = {
        "symbol": config.symbol,
        "interval": config.interval,
        "outputsize": config.outputsize,
        "timezone": config.timezone,
        "format": "JSON",
    }
    if start is not None and end is not None:
        params["start_date"] = start.strftime("%Y-%m-%dT%H:%M:%SZ")
        params["end_date"] = end.strftime("%Y-%m-%dT%H:%M:%SZ")
        params.pop("outputsize", None)

    for attempt in range(3):
        try:
            response = requests.get(
                "https://api.twelvedata.com/time_series",
                params=params,
                headers={"Authorization": f"apikey {key}"},
                timeout=config.timeout_seconds,
            )
            if response.status_code == 429 and attempt < 2:
                time.sleep(65)
                continue
            response.raise_for_status()
            payload = response.json()
        except RequestException as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            detail = ""
            resp = getattr(exc, "response", None)
            if resp is not None:
                try:
                    body = resp.json()
                    detail = str(body.get("message") or body.get("code") or "")
                except ValueError:
                    detail = ""
            suffix = f"; status={status}" if status is not None else ""
            if detail:
                suffix += f"; message={detail}"
            raise RuntimeError(
                f"TwelveData request failed: {exc.__class__.__name__}{suffix}"
            ) from None
        except ValueError:
            raise RuntimeError("TwelveData returned a non-JSON response") from None

        if payload.get("status") == "error" or "values" not in payload:
            raise RuntimeError(
                f"TwelveData error: {payload.get('message', 'invalid response')}"
            )
        rows = [
            {
                "timestamp": r["datetime"],
                "open": r["open"],
                "high": r["high"],
                "low": r["low"],
                "close": r["close"],
                "volume": r.get("volume"),
            }
            for r in payload["values"]
        ]
        return validate_ohlcv(pd.DataFrame(rows))

    raise RuntimeError("TwelveData rate limit persisted after retries")

def fetch_twelvedata(config: DataConfig, api_key: Optional[str] = None, days: int = 30) -> pd.DataFrame:
    key = api_key or os.getenv("TWELVEDATA_API_KEY")
    if not key:
        raise RuntimeError("TWELVEDATA_API_KEY is missing; no valid market data is available")
    if days < 1:
        raise ValueError("days must be >= 1")
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    chunks = []
    window = timedelta(days=10)
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + window, end)
        print(
            f"FARMANDEHAI_DATA_CHUNK_START={cursor.isoformat()}..{chunk_end.isoformat()}",
            flush=True,
        )
        chunk = _request(config, key, cursor, chunk_end)
        print(
            f"FARMANDEHAI_DATA_CHUNK_DONE={len(chunk)}",
            flush=True,
        )
        chunks.append(chunk)
        cursor = chunk_end + timedelta(minutes=5)
    if not chunks:
        raise RuntimeError("No market data returned")
    return validate_ohlcv(pd.concat(chunks, ignore_index=True))
