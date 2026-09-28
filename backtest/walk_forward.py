from __future__ import annotations
from backtest.engine import run_backtest
from backtest.metrics import metrics

def walk_forward(df, params, folds=4, train_ratio=0.5):
    if folds < 2:
        raise ValueError("folds must be >= 2")
    if not 0.25 <= train_ratio <= 0.8:
        raise ValueError("train_ratio must be between 0.25 and 0.8")
    n = len(df)
    if n < folds * 20:
        raise ValueError("not enough rows for walk-forward")
    first_oos = max(1, int(n * train_ratio))
    remaining = n - first_oos
    oos_size = remaining // folds
    results = []; combined = []
    for k in range(folds):
        oos_start = first_oos + k * oos_size
        oos_end = first_oos + (k + 1) * oos_size if k < folds - 1 else n
        train = df.iloc[:oos_start].reset_index(drop=True)
        oos = df.iloc[oos_start:oos_end].reset_index(drop=True)
        train_trades = run_backtest(train, params)
        oos_trades = run_backtest(oos, params)
        results.append({"fold":k+1,"train_rows":len(train),"oos_rows":len(oos),"train_metrics":metrics([t.r for t in train_trades]),"metrics":metrics([t.r for t in oos_trades]),"trades":oos_trades})
        combined.extend(oos_trades)
    return results, metrics([t.r for t in combined])
