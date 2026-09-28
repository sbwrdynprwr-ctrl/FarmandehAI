from __future__ import annotations
import argparse
from data.loader import DataConfig, fetch_twelvedata, load_csv
from strategy.strategy import StrategyParams
from backtest.engine import run_backtest
from backtest.metrics import metrics
from backtest.walk_forward import walk_forward
from reports.reporter import write_reports

PAPER=True; LIVE=False; REAL=False; NO_LOOKAHEAD=True; CLOSED_ONLY=True
if LIVE or REAL: raise RuntimeError("Safety lock violated")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--csv"); ap.add_argument("--days",type=int,default=30); ap.add_argument("--folds",type=int,default=4); ap.add_argument("--run-tests",action="store_true"); args=ap.parse_args()
    if args.run_tests:
        import subprocess,sys
        raise SystemExit(subprocess.call([sys.executable,"-m","pytest","-q"]))
    df=load_csv(args.csv) if args.csv else fetch_twelvedata(DataConfig(),days=args.days)
    params=StrategyParams(); trades=run_backtest(df,params); bm=metrics([t.r for t in trades]); wf,wfm=walk_forward(df,params,args.folds)
    status=("PROJECT COMPLETION: 85%\nPROJECT REMAINING: 15%\nVERSION: v0.1.4-checkpoint\nLIVE TRADING: OFF\nREAL ORDER: OFF\nPAPER TRADING: ON\nREAL HISTORICAL BACKTEST: EXECUTED ONLY IF TWELVEDATA_API_KEY WAS AVAILABLE\nVALIDATION: NOT CONFIRMED\nNEXT STEP: Review genuine-data Walk-Forward and robustness evidence before any deployment decision.")
    write_reports("artifacts",bm,[{"fold":r["fold"],"metrics":r["metrics"]} for r in wf],trades,status)
    print({"backtest":bm,"walk_forward":wfm,"validation":"NOT CONFIRMED"})
if __name__=="__main__": main()
