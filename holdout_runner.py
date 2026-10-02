import json
import os
import time
import traceback

print("FARMANDEHAI_BOOT=python_started", flush=True)
print("FARMANDEHAI_BUILD=post-0521-robustness", flush=True)
print(f"FARMANDEHAI_ENV=PAPER:{os.getenv('PAPER')} LIVE:{os.getenv('LIVE')} REAL:{os.getenv('REAL')}", flush=True)

try:
    from data.loader import DataConfig, fetch_twelvedata
    from backtest.research import independent_holdout, walk_forward_search, robustness, monte_carlo

    print("FARMANDEHAI_FETCH_START", flush=True)
    started = time.time()
    df = fetch_twelvedata(DataConfig(), days=90)
    print(f"DATA_ROWS={len(df)} FETCH_SECONDS={time.time() - started:.1f}", flush=True)

    holdout_ratio = float(os.getenv("HOLDOUT_RATIO", "0.10"))
    results = independent_holdout(df, holdout_ratio=holdout_ratio)
    for hypothesis, result in results.items():
        print(f"FORWARD_HOLDOUT={hypothesis}:{json.dumps(result, default=str)}", flush=True)
    print("FARMANDEHAI_FORWARD_HOLDOUT_DONE", flush=True)

    # Independent walk-forward robustness study. Each fold selects parameters
    # only from data available before that fold's OOS segment.
    for hypothesis in ("trend_filtered", "trend", "trend_regime", "mean_reversion_v2", "mean_reversion", "breakout", "pullback"):
        print(f"ROBUSTNESS_WF_START={hypothesis}", flush=True)
        started = time.time()
        folds, combined = walk_forward_search(df, folds=4, train_ratio=0.5, hypothesis=hypothesis)
        rb = robustness(df, folds)
        mc = monte_carlo(
            [t.r for fold in folds for t in fold["trades"]],
            simulations=int(os.getenv("MC_SIMULATIONS", "1000")),
            seed=42,
        )
        print(
            f"ROBUSTNESS_WF_SUMMARY={hypothesis}:"
            f"{json.dumps({'combined_oos': combined, 'folds': [{'fold': x['fold'], 'trades': x['oos_metrics']['trades'], 'total_r': x['oos_metrics']['total_r'], 'expectancy': x['oos_metrics']['expectancy'], 'profit_factor': x['oos_metrics']['profit_factor']} for x in folds], 'robustness': rb, 'monte_carlo': mc}, default=str)}",
            flush=True,
        )
        print(f"ROBUSTNESS_WF_DONE={hypothesis} SECONDS={time.time() - started:.1f}", flush=True)

    print("FARMANDEHAI_ROBUSTNESS_DONE", flush=True)
except BaseException as exc:
    print(f"FARMANDEHAI_ROBUSTNESS_ERROR={type(exc).__name__}:{exc}", flush=True)
    traceback.print_exc()
    raise
