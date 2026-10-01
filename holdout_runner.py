import json

from data.loader import DataConfig, fetch_twelvedata
from backtest.research import independent_holdout

print("FARMANDEHAI_HOLDOUT_START", flush=True)
df = fetch_twelvedata(DataConfig(), days=60)
print(f"DATA_ROWS={len(df)}", flush=True)
results = independent_holdout(df, holdout_ratio=0.20)
for hypothesis, result in results.items():
    print(f"INDEPENDENT_HOLDOUT={hypothesis}:{json.dumps(result, default=str)}", flush=True)
print("INDEPENDENT_HOLDOUT_DONE", flush=True)
