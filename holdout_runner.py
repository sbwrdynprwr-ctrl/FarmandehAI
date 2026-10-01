import json
import os
import time
import traceback

from data.loader import DataConfig, fetch_twelvedata
from backtest.research import independent_holdout

print("FARMANDEHAI_FORWARD_HOLDOUT_START", flush=True)
try:
    started = time.time()
    df = fetch_twelvedata(DataConfig(), days=90)
    print(f"DATA_ROWS={len(df)} FETCH_SECONDS={time.time() - started:.1f}", flush=True)
    holdout_ratio = float(os.getenv("HOLDOUT_RATIO", "0.10"))
    print(f"HOLDOUT_RATIO={holdout_ratio}", flush=True)
    results = independent_holdout(df, holdout_ratio=holdout_ratio)
    for hypothesis, result in results.items():
        print(f"FORWARD_HOLDOUT={hypothesis}:{json.dumps(result, default=str)}", flush=True)
    print("FARMANDEHAI_FORWARD_HOLDOUT_DONE", flush=True)
except BaseException as exc:
    print(f"FARMANDEHAI_FORWARD_HOLDOUT_ERROR={type(exc).__name__}:{exc}", flush=True)
    traceback.print_exc()
    raise
