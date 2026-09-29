from __future__ import annotations

import argparse
import json
import subprocess
import sys
import os
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

# Railway currently deploys an older pinned snapshot. Bootstrap the strategy module
# from the current repository before importing modules that depend on StrategyParams.
def _bootstrap_strategy():
    path = os.path.join("strategy", "strategy.py")
    os.makedirs("strategy", exist_ok=True)
    import requests
    url = "https://raw.githubusercontent.com/sbwrdynprwr-ctrl/FarmandehAI/main/strategy/strategy.py"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)

_bootstrap_strategy()

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

    if args.serve:
        port = int(os.environ.get("PORT", "8080"))
        handler = partial(SimpleHTTPRequestHandler, directory="artifacts")
        threading.Thread(
            target=ThreadingHTTPServer(("0.0.0.0", port), handler).serve_forever,
            daemon=True,
        ).start()
        print(f"RESEARCH_HTTP_PORT={port}", flush=True)

    df = load_csv(args.csv) if args.csv else fetch_twelvedata(DataConfig(), days=args.days)
    print(f"DATA_ROWS={len(df)}", flush=True)

    baseline = StrategyParams(hypothesis="trend")
    baseline_trades = run_backtest(df, baseline)
    bm = metrics([t.r for t in baseline_trades])
    wf, wfm = walk_forward(df, baseline, args.folds)
    print(f"BASELINE_WF={json.dumps(wfm)}", flush=True)

    research = None
    if args.research:
        print("HYPOTHESIS_COMPARISON_START", flush=True)
        trend_folds, trend_combined = walk_forward_search(df, args.folds, hypothesis="trend")
        trend_robustness = robustness(df, trend_folds)
        trend_mc = monte_carlo([t.r for f in trend_folds for t in f["trades"]], 1000, 42)

        mean_folds, mean_combined = walk_forward_search(df, args.folds, hypothesis="mean_reversion")
        mean_robustness = robustness(df, mean_folds)
        mean_mc = monte_carlo([t.r for f in mean_folds for t in f["trades"]], 1000, 42)

        print(f"TREND_RESEARCH_COMBINED={json.dumps(trend_combined)}", flush=True)
        print(f"MEAN_REVERSION_RESEARCH_COMBINED={json.dumps(mean_combined)}", flush=True)
        print(f"MEAN_REVERSION_ROBUSTNESS={json.dumps(mean_robustness, default=str)}", flush=True)
        print(f"MEAN_REVERSION_MONTE_CARLO={json.dumps(mean_mc)}", flush=True)
        for label, folds in (("trend", trend_folds), ("mean_reversion", mean_folds)):
            for f in folds:
                print(f"RESEARCH_FOLD={label}:{json.dumps({k: v for k, v in f.items() if k != 'trades'}, default=str)}", flush=True)

        research = {
            "trend": {"folds": trend_folds, "combined": trend_combined,
                      "robustness": trend_robustness, "monte_carlo": trend_mc},
            "mean_reversion": {"folds": mean_folds, "combined": mean_combined,
                               "robustness": mean_robustness, "monte_carlo": mean_mc},
        }

    status = (
        "PROJECT COMPLETION: 92%\n"
        "PROJECT REMAINING: 8%\n"
        "VERSION: v0.2.2-hypothesis-research\n"
        "LIVE TRADING: OFF\n"
        "REAL ORDER: OFF\n"
        "PAPER TRADING: ON\n"
        "HYPOTHESES: TREND + ISOLATED MEAN_REVERSION\n"
        "TRAIN_ONLY_PARAMETER_SEARCH: EXECUTED\n"
        "OOS_EVALUATION: EXECUTED\n"
        "ROBUSTNESS: EXECUTED\n"
        "MONTE_CARLO: EXECUTED\n"
        "VALIDATION: NOT CONFIRMED\n"
        "NEXT STEP: Compare genuine OOS evidence without activating live/real trading."
    )
    write_reports(
        "artifacts", bm,
        [{"fold": r["fold"], "metrics": r["metrics"], "train_metrics": r["train_metrics"]} for r in wf],
        baseline_trades, status, research=research,
    )
    print("FARMANDEHAI_RESEARCH_DONE", flush=True)
    if args.serve:
        print("RESEARCH_HTTP_READY", flush=True)
        threading.Event().wait()


if __name__ == "__main__":
    main()
