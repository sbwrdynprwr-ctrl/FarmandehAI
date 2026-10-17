import json
import os
import time
import traceback

print("FARMANDEHAI_BOOT=python_started", flush=True)
print("FARMANDEHAI_BUILD=cost-robust-search-v2", flush=True)
print(f"FARMANDEHAI_ENV=PAPER:{os.getenv('PAPER')} LIVE:{os.getenv('LIVE')} REAL:{os.getenv('REAL')}", flush=True)

try:
    from data.loader import DataConfig, load_research_data
    from backtest.research import independent_holdout, walk_forward_search, robustness, monte_carlo
    from backtest.gates import validation_gate_details
    from evaluation.shadow_ab import compare_ab_by_timestamp

    print("FARMANDEHAI_FETCH_START", flush=True)
    started = time.time()
    research_days = int(os.getenv("RESEARCH_DAYS", "180"))
    print(f"FARMANDEHAI_RESEARCH_DAYS={research_days}", flush=True)
    df = load_research_data(DataConfig(), days=research_days)
    print(f"DATA_ROWS={len(df)} FETCH_SECONDS={time.time() - started:.1f}", flush=True)

    holdout_ratio = float(os.getenv("HOLDOUT_RATIO", "0.10"))
    checkpoint_dir = os.getenv("FARMANDEHAI_CHECKPOINT_DIR", "artifacts/checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    results = independent_holdout(
        df,
        holdout_ratio=holdout_ratio,
        checkpoint_path=os.path.join(checkpoint_dir, "holdout_checkpoint.json"),
    )
    for hypothesis, result in results.items():
        print(f"FORWARD_HOLDOUT={hypothesis}:{json.dumps(result, default=str)}", flush=True)
    print("FARMANDEHAI_FORWARD_HOLDOUT_DONE", flush=True)

    validation = {}
    validation_progress_path = os.path.join(checkpoint_dir, "validation_partial.json")
    validation_checkpoint_version = 2
    if os.path.exists(validation_progress_path):
        try:
            with open(validation_progress_path, "r", encoding="utf-8") as fp:
                saved_validation = json.load(fp)
            if (
                isinstance(saved_validation, dict)
                and saved_validation.get("_checkpoint_version") == validation_checkpoint_version
            ):
                validation.update(
                    {k: v for k, v in saved_validation.items() if k != "_checkpoint_version"}
                )
                print(f"VALIDATION_PARTIAL_RESUME={len(validation)}", flush=True)
            else:
                print(
                    "VALIDATION_PARTIAL_INVALIDATED="
                    f"expected:{validation_checkpoint_version}",
                    flush=True,
                )
        except (OSError, ValueError, TypeError):
            print("VALIDATION_PARTIAL_RESUME=0", flush=True)
    holdout_report = results
    for hypothesis in ("trend_filtered", "trend", "trend_regime", "mean_reversion_v2", "mean_reversion_rr", "mean_reversion_costaware", "mean_reversion_robust", "mean_reversion_regime", "mean_reversion_band", "mean_reversion_v3", "mean_reversion", "breakout", "pullback"):
        existing = validation.get(hypothesis)
        if (
            isinstance(existing, dict)
            and existing.get("combined") is not None
            and existing.get("robustness") is not None
            and existing.get("monte_carlo") is not None
            and existing.get("shadow_ab") is not None
        ):
            print(f"ROBUSTNESS_WF_RESUME_COMPLETE={hypothesis}", flush=True)
            continue

        print(f"ROBUSTNESS_WF_START={hypothesis}", flush=True)
        started = time.time()
        folds, combined = walk_forward_search(
            df,
            folds=4,
            train_ratio=0.5,
            hypothesis=hypothesis,
            checkpoint_path=os.path.join(checkpoint_dir, f"wf_{hypothesis}.json"),
        )
        print(f"ROBUSTNESS_METRICS_START={hypothesis} FOLDS={len(folds)}", flush=True)
        try:
            rb = robustness(df, folds)
        except BaseException as exc:
            print(
                f"ROBUSTNESS_METRICS_ERROR={hypothesis} "
                f"{type(exc).__name__}:{exc}",
                flush=True,
            )
            raise
        print(f"ROBUSTNESS_METRICS_DONE={hypothesis}", flush=True)

        # Compare selected parameters with the fixed baseline on identical OOS
        # timestamp slots. This is offline analysis only and never enables orders.
        first_oos = max(1, int(len(df) * 0.5))
        oos_size = (len(df) - first_oos) // len(folds)
        evaluation_timestamps = []
        baseline_trades = []
        candidate_trades = []
        for fold in folds:
            fold_no = int(fold["fold"])
            oos_start = first_oos + (fold_no - 1) * oos_size
            oos_end = first_oos + fold_no * oos_size if fold_no < len(folds) else len(df)
            # run_backtest_window starts signal evaluation at oos_start - 1
            # so that a signal on the preceding bar may enter at the first OOS
            # bar. Trade.timestamp records that signal bar (ts[i]), therefore
            # align A/B slots to the exact signal timestamps used by the engine.
            # The half-open slice has the same length as this fold's OOS window
            # and adjacent folds remain non-overlapping.
            evaluation_timestamps.extend(
                str(value)
                for value in df["timestamp"].to_numpy()[oos_start - 1:oos_end - 1]
            )
            baseline_trades.extend(fold["baseline_trades"])
            candidate_trades.extend(fold["trades"])
        shadow_ab = compare_ab_by_timestamp(
            evaluation_timestamps,
            baseline_trades,
            candidate_trades,
            min_samples=100,
        )
        print(
            f"SHADOW_AB_RESULT={hypothesis}:"
            f"{json.dumps(shadow_ab, default=str)}",
            flush=True,
        )

        print(f"MONTE_CARLO_START={hypothesis}", flush=True)
        mc = monte_carlo(
            [t.r for fold in folds for t in fold["trades"]],
            simulations=int(os.getenv("MC_SIMULATIONS", "1000")),
            seed=42,
        )
        print(f"MONTE_CARLO_DONE={hypothesis}", flush=True)
        print(
            f"ROBUSTNESS_WF_SUMMARY={hypothesis}:"
            f"{json.dumps({'combined_oos': combined, 'folds': [{'fold': x['fold'], 'trades': x['oos_metrics']['trades'], 'total_r': x['oos_metrics']['total_r'], 'expectancy': x['oos_metrics']['expectancy'], 'profit_factor': x['oos_metrics']['profit_factor']} for x in folds], 'robustness': rb, 'monte_carlo': mc}, default=str)}",
            flush=True,
        )
        validation[hypothesis] = {
            "hypothesis": hypothesis,
            "combined": combined,
            "robustness": rb,
            "monte_carlo": mc,
            "shadow_ab": shadow_ab,
            "selection_statuses": [
                fold.get("selection", {}).get("selection_status", "unknown")
                for fold in folds
            ],
            "selection_diagnostics": [
                {
                    "fold": fold.get("fold"),
                    **fold.get("selection", {}).get("selection_diagnostics", {}),
                }
                for fold in folds
            ],
        }
        with open(validation_progress_path, "w", encoding="utf-8") as fp:
            json.dump(
                {"_checkpoint_version": validation_checkpoint_version, **validation},
                fp,
                default=str,
            )
        print(f"ROBUSTNESS_WF_DONE={hypothesis} SECONDS={time.time() - started:.1f}", flush=True)

    approved = []
    for hypothesis, report in validation.items():
        gate_details = validation_gate_details(report)
        validation[hypothesis]["gate_pass"] = gate_details["pass"]
        validation[hypothesis]["gate_details"] = gate_details
        if gate_details["pass"]:
            approved.append(hypothesis)

    validation_payload = {
        "build": os.getenv("FARMANDEHAI_BUILD", "unknown"),
        "research_days": research_days,
        "holdout_ratio": holdout_ratio,
        "approved": approved,
        "live_enabled": False,
        "real_enabled": False,
        "holdout": holdout_report,
        "validation": validation,
    }
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/validation.json", "w", encoding="utf-8") as fp:
        json.dump(validation_payload, fp, indent=2)
    print("FARMANDEHAI_VALIDATION=" + json.dumps(validation_payload), flush=True)
    print("FARMANDEHAI_ROBUSTNESS_DONE", flush=True)
except BaseException as exc:
    print(f"FARMANDEHAI_ROBUSTNESS_ERROR={type(exc).__name__}:{exc}", flush=True)
    traceback.print_exc()
    raise
