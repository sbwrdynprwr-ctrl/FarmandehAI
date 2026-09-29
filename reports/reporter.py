import csv
import json
from pathlib import Path


def _clean_research(research):
    return {
        name: {
            "folds": [{k: v for k, v in f.items() if k != "trades"} for f in data.get("folds", [])],
            "combined": data.get("combined"),
            "robustness": data.get("robustness"),
            "monte_carlo": data.get("monte_carlo"),
        }
        for name, data in research.items()
    }


def write_reports(outdir, backtest_metrics, wf_results, trades, status, research=None):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"backtest": backtest_metrics, "walk_forward": wf_results}
    if research is not None:
        payload["research"] = _clean_research(research)
    (out / "metrics.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    lines = ["WALK-FORWARD REPORT", ""]
    for r in wf_results:
        lines.append(f"Fold {r['fold']}: train={r.get('train_metrics')} oos={r['metrics']}")
    (out / "walk_forward_report.txt").write_text("\n".join(lines), encoding="utf-8")
    (out / "backtest_report.txt").write_text(json.dumps(backtest_metrics, indent=2), encoding="utf-8")

    with (out / "trades.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "side", "entry", "stop_loss", "take_profit", "exit", "result", "r", "reason"])
        for t in trades:
            w.writerow([t.timestamp, t.side, t.entry, t.stop_loss, t.take_profit, t.exit, t.result, t.r, t.reason])

    if research is not None:
        (out / "research_report.json").write_text(
            json.dumps(_clean_research(research), indent=2, default=str),
            encoding="utf-8",
        )

    (out / "project_status.txt").write_text(status, encoding="utf-8")
