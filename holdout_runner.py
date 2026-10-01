import json

from data.loader import DataConfig, fetch_twelvedata
from backtest.research import independent_holdout

# Fresh forward holdout: use a longer history so the final 10% is later
# Triggered after the training-screening performance fix; logic remains unchanged.
# than the previously inspected 60-day holdout.
print("FARMANDEHAI_FORWARD_HOLDOUT_START", flush=True)
df = fetch_twelvedata(DataConfig(), days=90)
print(f"DATA_ROWS={len(df)}", flush=True)
results = independent_holdout(df, holdout_ratio=0.10)
for hypothesis, result in results.items():
    print(f"FORWARD_HOLDOUT={hypothesis}:{json.dumps(result, default=str)}", flush=True)
print("FARMANDEHAI_FORWARD_HOLDOUT_DONE", flush=True)
