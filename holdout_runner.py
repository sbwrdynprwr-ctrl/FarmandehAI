import json
import os
import time
import traceback

print("FARMANDEHAI_BOOT=python_started", flush=True)
print("FARMANDEHAI_BUILD=post-1002-180d-validation", flush=True)
print(f"FARMANDEHAI_ENV=PAPER:{os.getenv('PAPER')} LIVE:{os.getenv('LIVE')} REAL:{os.getenv('REAL')}", flush=True)

try:
    from data.loader import DataConfig, fetch_twelvedata
    from backtest.research import independent_holdout, walk_forward_search, robustness, monte_carlo

    print("FARMANDEHAI_FETCH_START", flush=True)
    started = time.time()
    research_days = int(os.getenv("RESEARCH_DAYS", "180"))
    print(f"FARMANDEHAI_RESEARCH_DAYS={research_days}", flush=True)
    df = fetch_twelvedata(DataConfig(), days=research_days)
    print(f"DATA_ROWS={len(df)} FETCH_SECONDS={time.time() - started:.1f}", flush=True)

    holdout_ratio = float(os.getenv("HOLDOUT_RATIO", "0.10"))
    results = independent_holdout(df, holdout_ratio=holdout_ratio)
    for hypothesis, result in results.items():
        print(f"FORWARD_HOLDOUT={hypothesis}:{json.dumps(result, default=str)}", flush=True)
    print("FARMANDEHAI_FORWARD_HOLDOUT_DONE", flush=True)

    validation = {}
    for hypothesis in ("trend_filtered", "trend", "trend_regime", "mean_reversion_v2", "mean_reversion_rr", "mean_reversion_costaware", "mean_reversion_v3", "mean_reversion", "breakout", "pullback"):
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
        validation[hypothesis] = {"combined": combined, "robustness": rb, "monte_carlo": mc}
        print(f"ROBUSTNESS_WF_DONE={hypothesis} SECONDS={time.time() - started:.1f}", flush=True)

    approved = []
    for hypothesis, report in validation.items():
        combined = report["combined"]
        folds = report["robustness"].get("folds", [])
        positive_folds = sum(1 for x in folds if x["selected_oos"]["total_r"] > 0)
        neighbor_rates = [x["neighbor_positive_rate"] for x in folds if x["neighbor_count"]]
        spread_5bps = sum(
            x["metrics"]["total_r"]
            for x in report["robustness"].get("spread_sensitivity", [])
            if x["spread"] == 0.00005
        )
        terminal_r = report["monte_carlo"].get("terminal_r", 0.0)
        gate = (
            combined.get("trades", 0) >= 100
            and combined.get("total_r", 0.0) > 0.0
            and combined.get("expectancy", 0.0) > 0.0
            and positive_folds >= 3
            and (min(neighbor_rates) if neighbor_rates else 0.0) >= 0.50
            and spread_5bps > 0.0
            and terminal_r > 0.0
        )
        validation[hypothesis]["gate_pass"] = gate
        if gate:
            approved.append(hypothesis)

    validation_payload = {"approved": approved, "live_enabled": False, "real_enabled": False}
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/validation.json", "w", encoding="utf-8") as fp:
        json.dump(validation_payload, fp, indent=2)
    print("FARMANDEHAI_VALIDATION=" + json.dumps(validation_payload), flush=True)
    print("FARMANDEHAI_ROBUSTNESS_DONE", flush=True)
except BaseException as exc:
    print(f"FARMANDEHAI_ROBUSTNESS_ERROR={type(exc).__name__}:{exc}", flush=True)
    traceback.print_exc()
    raise
