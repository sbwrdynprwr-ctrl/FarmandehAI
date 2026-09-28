from backtest.metrics import metrics

def test_metrics_order_and_streak():
    m=metrics([2,-1,-1,2]); assert m["trades"]==4; assert m["wins"]==2; assert m["max_loss_streak"]==2; assert m["max_drawdown"]==2
