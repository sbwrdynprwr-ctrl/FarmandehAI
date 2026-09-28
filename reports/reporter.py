import json
from pathlib import Path

def write_reports(outdir, backtest_metrics, wf_results, trades, status):
    out=Path(outdir); out.mkdir(parents=True, exist_ok=True)
    (out/"metrics.json").write_text(json.dumps({"backtest":backtest_metrics,"walk_forward":wf_results},indent=2), encoding="utf-8")
    lines=["WALK-FORWARD REPORT", ""]
    for r in wf_results: lines.append(f"Fold {r['fold']}: {r['metrics']}")
    (out/"walk_forward_report.txt").write_text("\n".join(lines), encoding="utf-8")
    (out/"backtest_report.txt").write_text(json.dumps(backtest_metrics,indent=2), encoding="utf-8")
    with (out/"trades.csv").open("w",encoding="utf-8") as f:
        f.write("timestamp,side,entry,stop_loss,take_profit,exit,result,r,reason\n")
        for t in trades: f.write(f"{t.timestamp},{t.side},{t.entry},{t.stop_loss},{t.take_profit},{t.exit},{t.result},{t.r},{t.reason}\n")
    (out/"project_status.txt").write_text(status,encoding="utf-8")
