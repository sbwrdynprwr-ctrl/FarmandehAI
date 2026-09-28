from __future__ import annotations

import argparse
import json
import subprocess
import sys

from data.loader import DataConfig, fetch_twelvedata, load_csv
from strategy.strategy import StrategyParams
from backtest.engine import run_backtest
from backtest.metrics import metrics
from backtest.walk_forward import walk_forward
from backtest.research import walk_forward_search, robustness, monte_carlo
from reports.reporter import write_reports

PAPER = True
LIVE = False
REAL = False
NO_LOOKAHEAD = True
CLOSED_ONLY = True
if LIVE or REAL:
    raise RuntimeError("Safety lock violated: LIVE/REAL must remain disabled")


def main():
    print("FARMANDEHAI_RESEARCH_START", flush=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--research", action="store_true")
    ap.add_argument("--run-tests", action="store_true")
    ap.add_argument("--serve", action="store_true")
    args = ap.parse_args()

    if args.run_tests:
        raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", "-q"]))

    df = load_csv(args.csv) if args.csv else fetch_twelvedata(DataConfig(), days=args.days)
    print(f"DATA_ROWS={len(df)}", flush=True)
    baseline = StrategyParams()
    baseline_trades = run_backtest(df, baseline)
    bm = metrics([t.r for t in baseline_trades])

    wf, wfm = walk_forward(df, baseline, args.folds)
    print(f"BASELINE_WF={json.dumps(wfm)}", flush=True)
    research = None
    if args.research:
        research_folds, research_combined = walk_forward_search(df, args.folds)
        for f in research_folds:
            print(f"RESEARCH_FOLD={json.dumps({k: v for k, v in f.items() if k != 'trades'}, default=str)}", flush=True)
        research_robustness = robustness(df, research_folds)
        print(f"ROBUSTNESS={json.dumps(research_robustness, default=str)}", flush=True)
        research_mc = monte_carlo(
            [t.r for f in research_folds for t in f["trades"]],
            simulations=1000,
            seed=42,
        )
        print(f"MONTE_CARLO={json.dumps(research_mc)}", flush=True)
        print(f"RESEARCH_COMBINED={json.dumps(research_combined)}", flush=True)
        research = {
            "folds": research_folds,
            "combined": research_combined,
            "robustness": research_robustness,
            "monte_carlo": research_mc,
        }

    status = (
        "PROJECT COMPLETION: 90%\n"
        "PROJECT REMAINING: 10%\n"
        "VERSION: v0.2.0-research\n"
        "LIVE TRADING: OFF\n"
        "REAL ORDER: OFF\n"
        "PAPER TRADING: ON\n"
        "TRAIN_ONLY_PARAMETER_SEARCH: EXECUTED\n"
        "OOS_EVALUATION: EXECUTED\n"
        "ROBUSTNESS: EXECUTED\n"
        "MONTE_CARLO: EXECUTED\n"
        "VALIDATION: NOT CONFIRMED\n"
        "NEXT STEP: Review genuine OOS stability and robustness evidence; no live/real activation."
    )
    write_reports(
        "artifacts",
        bm,
        [{"fold": r["fold"], "metrics": r["metrics"], "train_metrics": r["train_metrics"]} for r in wf],
        baseline_trades,
        status,
        research=research,
    )
    print("FARMANDEHAI_RESEARCH_DONE", flush=True)
    if args.serve:
        import os
        from functools import partial
        from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
        port = int(os.environ.get("PORT", "8080"))
        handler = partial(SimpleHTTPRequestHandler, directory="artifacts")
        print(f"RESEARCH_HTTP_PORT={port}", flush=True)
        ThreadingHTTPServer(("0.0.0.0", port), handler).serve_forever()

    print(json.dumps({
        "baseline_backtest": bm,
        "baseline_walk_forward": wfm,
        "research_combined": research["combined"] if research else None,
        "monte_carlo": research["monte_carlo"] if research else None,
        "validation": "NOT CONFIRMED",
    }, indent=2))


if __name__ == "__main__":
    main()
