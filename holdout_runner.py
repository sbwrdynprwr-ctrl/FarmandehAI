import json
import os
import time
import traceback

print("FARMANDEHAI_BOOT=python_started", flush=True)
print(f"FARMANDEHAI_ENV=PAPER:{os.getenv('PAPER')} LIVE:{os.getenv('LIVE')} REAL:{os.getenv('REAL')}", flush=True)

try:
    print("FARMANDEHAI_BOOT=import_data_loader", flush=True)
    from data.loader import DataConfig, fetch_twelvedata
    print("FARMANDEHAI_BOOT=import_data_loader_ok", flush=True)

    print("FARMANDEHAI_BOOT=import_research", flush=True)
    from backtest.research import independent_holdout
    print("FARMANDEHAI_BOOT=import_research_ok", flush=True)

    print("FARMANDEHAI_FORWARD_HOLDOUT_START", flush=True)
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
