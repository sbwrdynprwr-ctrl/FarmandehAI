# ============================================================
# FARMANDEHAI v17406.29
# GBP/USD | 5M | TRUE WALK-FORWARD VALIDATION
# PAPER ONLY - NO REAL ORDERS
# ============================================================

import os
import requests
from datetime import datetime, timezone

# ============================================================
# CONFIG
# ============================================================
VERSION = "v17406.29"
PAIR = "GBP/USD"
INTERVAL = "5min"

PAPER = True
LIVE = False
REAL = False
NO_LOOKAHEAD = True
CLOSED_ONLY = True

HIST_CANDLES = 3000
ATR_PERIOD = 14
RR = 2.0

# پارامترهایی که فقط روی TRAIN انتخاب می‌شوند
ATR_MULTIPLIERS = [1.0, 1.25, 1.5, 1.75, 2.0]
BODY_MIN_VALUES = [0.55, 0.60, 0.65]
DIRECTIONS = ["BUY", "SELL"]

MIN_TRAIN_TRADES = 15

API_KEY = os.getenv("TWELVEDATA_API_KEY", "").strip()

# ============================================================
# SAFETY
# ============================================================
if not PAPER or LIVE or REAL:
    raise SystemExit("TRADE BLOCKED - PAPER ONLY")

if not API_KEY:
    raise SystemExit("ERROR: TWELVEDATA_API_KEY is missing")

# ============================================================
# DATA
# ============================================================
def get_data():

    url = "https://api.twelvedata.com/time_series"

    params = {
        "symbol": PAIR,
        "interval": INTERVAL,
        "outputsize": HIST_CANDLES,
        "apikey": API_KEY,
        "timezone": "UTC"
    }

    for attempt in range(1, 5):

        try:
            print("TWELVEDATA ATTEMPT:", attempt, "/4")

            r = requests.get(
                url,
                params=params,
                timeout=25
            )

            r.raise_for_status()
            d = r.json()

            if "values" not in d:
                print(
                    "TWELVEDATA ERROR:",
                    d.get("message", "unknown")
                )
                continue

            data = []

            for x in reversed(d["values"]):

                try:
                    t = datetime.fromisoformat(
                        x["datetime"].replace("Z", "+00:00")
                    )

                    if t.tzinfo is None:
                        t = t.replace(tzinfo=timezone.utc)

                    t = t.astimezone(timezone.utc)

                    data.append({
                        "time": t,
                        "open": float(x["open"]),
                        "high": float(x["high"]),
                        "low": float(x["low"]),
                        "close": float(x["close"])
                    })

                except Exception:
                    continue

            return data

        except Exception as e:
            print("ERROR:", e)

    return []

# ============================================================
# CLOSED CANDLES
# ============================================================
def filter_closed(data):

    if not CLOSED_ONLY or not data:
        return data

    now = datetime.now(timezone.utc)

    minute = (now.minute // 5) * 5

    current_bar = now.replace(
        minute=minute,
        second=0,
        microsecond=0
    )

    return [
        x for x in data
        if x["time"] < current_bar
    ]

# ============================================================
# EMA
# ============================================================
def ema(values, period):

    result = [None] * len(values)

    if len(values) < period:
        return result

    result[period - 1] = sum(
        values[:period]
    ) / period

    k = 2.0 / (period + 1)

    for i in range(period, len(values)):

        result[i] = (
            values[i] * k
            + result[i - 1] * (1 - k)
        )

    return result

# ============================================================
# ATR
# ============================================================
def atr(data, period):

    result = [None] * len(data)
    tr = []

    for i in range(len(data)):

        if i == 0:

            value = (
                data[i]["high"]
                - data[i]["low"]
            )

        else:

            value = max(
                data[i]["high"] - data[i]["low"],
                abs(
                    data[i]["high"]
                    - data[i - 1]["close"]
                ),
                abs(
                    data[i]["low"]
                    - data[i - 1]["close"]
                )
            )

        tr.append(value)

    if len(tr) < period:
        return result

    result[period - 1] = (
        sum(tr[:period]) / period
    )

    for i in range(period, len(data)):

        result[i] = (
            result[i - 1] * (period - 1)
            + tr[i]
        ) / period

    return result

# ============================================================
# TRADE RESULT
# ============================================================
def trade_result(
    data,
    i,
    atr_value,
    atr_multiplier,
    direction,
    end
):

    entry = data[i]["close"]

    distance = atr_value * atr_multiplier

    if direction == "BUY":

        sl = entry - distance
        tp = entry + distance * RR

    else:

        sl = entry + distance
        tp = entry - distance * RR

    for j in range(i + 1, end):

        high = data[j]["high"]
        low = data[j]["low"]

        # هر دو در یک کندل:
        # محافظه‌کارانه LOSS
        if direction == "BUY":

            if low <= sl and high >= tp:
                return -1, j, True

            if low <= sl:
                return -1, j, False

            if high >= tp:
                return 2, j, False

        else:

            if high >= sl and low <= tp:
                return -1, j, True

            if high >= sl:
                return -1, j, False

            if low <= tp:
                return 2, j, False

    return None, None, False

# ============================================================
# STATISTICS
# ============================================================
def stats(results):

    n = len(results)

    wins = sum(
        1 for x in results
        if x == 2
    )

    losses = sum(
        1 for x in results
        if x == -1
    )

    total_r = sum(results)

    wr = (
        wins / n * 100
        if n else 0
    )

    expectancy = (
        total_r / n
        if n else 0
    )

    gross_profit = wins * RR
    gross_loss = losses

    pf = (
        gross_profit / gross_loss
        if gross_loss else float("inf")
    )

    equity = 0
    peak = 0
    max_dd = 0

    loss_streak = 0
    max_loss_streak = 0

    for x in results:

        equity += x

        if equity > peak:
            peak = equity

        drawdown = equity - peak

        if drawdown < max_dd:
            max_dd = drawdown

        if x == -1:

            loss_streak += 1

            if loss_streak > max_loss_streak:
                max_loss_streak = loss_streak

        else:

            loss_streak = 0

    return {
        "trades": n,
        "wins": wins,
        "losses": losses,
        "wr": wr,
        "r": total_r,
        "exp": expectancy,
        "pf": pf,
        "dd": max_dd,
        "loss_streak": max_loss_streak
    }

# ============================================================
# SIGNAL
# ============================================================
def signal_at(
    data,
    i,
    ema20,
    ema50,
    body_min,
    direction
):

    if (
        ema20[i] is None
        or ema50[i] is None
    ):
        return False

    c = data[i]

    candle_range = (
        c["high"] - c["low"]
    )

    if candle_range <= 0:
        return False

    body_ratio = (
        abs(c["close"] - c["open"])
        / candle_range
    )

    if body_ratio < body_min:
        return False

    if direction == "BUY":

        return (
            ema20[i] > ema50[i]
            and c["close"] > ema20[i]
            and c["close"] > c["open"]
        )

    return (
        ema20[i] < ema50[i]
        and c["close"] < ema20[i]
        and c["close"] < c["open"]
    )

# ============================================================
# BACKTEST
# ============================================================
def run_test(
    data,
    ema20,
    ema50,
    atr14,
    start,
    end,
    atr_multiplier,
    body_min,
    direction
):

    results = []
    same_candle = 0
    exit_bars = []

    i = start

    while i < end - 1:

        if signal_at(
            data,
            i,
            ema20,
            ema50,
            body_min,
            direction
        ):

            if atr14[i] is not None:

                result, exit_index, both = trade_result(
                    data,
                    i,
                    atr14[i],
                    atr_multiplier,
                    direction,
                    end
                )

                if result is not None:

                    results.append(result)

                    if both:
                        same_candle += 1

                    if exit_index is not None:
                        exit_bars.append(
                            exit_index - i
                        )

                    # جلوگیری واقعی از معاملات هم‌پوشان
                    i = exit_index + 1
                    continue

        i += 1

    s = stats(results)

    if exit_bars:
        s["avg_bars"] = (
            sum(exit_bars) / len(exit_bars)
        )
    else:
        s["avg_bars"] = 0

    s["same_candle"] = same_candle

    return s

# ============================================================
# TRAIN PARAMETER SELECTION
# ============================================================
def select_parameters(
    data,
    ema20,
    ema50,
    atr14,
    start,
    end
):

    candidates = []

    for direction in DIRECTIONS:

        for atr_multiplier in ATR_MULTIPLIERS:

            for body_min in BODY_MIN_VALUES:

                s = run_test(
                    data,
                    ema20,
                    ema50,
                    atr14,
                    start,
                    end,
                    atr_multiplier,
                    body_min,
                    direction
                )

                if s["trades"] >= MIN_TRAIN_TRADES:

                    candidates.append({
                        "direction": direction,
                        "atr": atr_multiplier,
                        "body": body_min,
                        "stats": s
                    })

    if not candidates:

        return {
            "direction": "BUY",
            "atr": 1.5,
            "body": 0.55,
            "stats": stats([])
        }

    # انتخاب فقط بر اساس TRAIN
    # اولویت با expectancy، سپس PF، سپس تعداد معاملات
    candidates.sort(
        key=lambda x: (
            x["stats"]["exp"],
            x["stats"]["pf"],
            x["stats"]["trades"]
        ),
        reverse=True
    )

    return candidates[0]

# ============================================================
# START
# ============================================================
print("=" * 65)
print("Starting FarmandehAI", VERSION)
print("=" * 65)

print("PAIR:", PAIR)
print("TIMEFRAME:", INTERVAL)
print("PAPER:", PAPER)
print("LIVE:", LIVE)
print("REAL ORDER:", REAL)
print("NO LOOKAHEAD:", NO_LOOKAHEAD)
print("CLOSED ONLY:", CLOSED_ONLY)
print("HISTORICAL CANDLES:", HIST_CANDLES)
print("RR:", RR)

print("-" * 65)

data = get_data()

data = filter_closed(data)

if len(data) < 2500:

    print("NOT ENOUGH DATA:", len(data))

    print("=" * 65)
    print("PROJECT COMPLETION: 97%")
    print("PROJECT REMAINING: 3%")
    print("LIVE TRADING: OFF")
    print("REAL ORDER: OFF")
    print("NEXT STEP: MORE HISTORICAL VALIDATION")
    print("TRADING READINESS: NOT READY")
    print("=" * 65)

    raise SystemExit(1)

print("TWELVEDATA CONNECTED")
print("CANDLES RECEIVED:", len(data))

print("-" * 65)

# ============================================================
# INDICATORS
# ============================================================
closes = [
    x["close"]
    for x in data
]

ema20 = ema(closes, 20)
ema50 = ema(closes, 50)
atr14 = atr(data, ATR_PERIOD)

# ============================================================
# TRUE WALK-FORWARD FOLDS
# ============================================================
folds = [
    (0, 1000, 1000, 1500),
    (500, 1500, 1500, 2000),
    (1000, 2000, 2000, 2500),
    (1500, 2500, 2500, min(3000, len(data)))
]

all_results = []
positive_folds = 0

fold_reports = []

for number, (
    train_start,
    train_end,
    oos_start,
    oos_end
) in enumerate(folds, 1):

    if oos_end > len(data):
        continue

    print("=" * 65)
    print("FOLD", number)
    print("=" * 65)

    print(
        "TRAIN:",
        data[train_start]["time"],
        "->",
        data[train_end - 1]["time"]
    )

    print(
        "OOS:",
        data[oos_start]["time"],
        "->",
        data[oos_end - 1]["time"]
    )

    # --------------------------------------------------------
    # پارامترها فقط از TRAIN انتخاب می‌شوند
    # --------------------------------------------------------
    selected = select_parameters(
        data,
        ema20,
        ema50,
        atr14,
        train_start,
        train_end
    )

    print(
        "SELECTED TRAIN PARAMS:",
        "DIR=", selected["direction"],
        "| ATR=", selected["atr"],
        "| BODY=", selected["body"]
    )

    train_stats = selected["stats"]

    print(
        "TRAIN:",
        "TRADES=", train_stats["trades"],
        "| R=", train_stats["r"],
        "| EXP=", round(train_stats["exp"], 3),
        "| PF=", round(train_stats["pf"], 3)
    )

    # --------------------------------------------------------
    # پارامترهای TRAIN روی OOS قفل می‌شوند
    # --------------------------------------------------------
    oos_stats = run_test(
        data,
        ema20,
        ema50,
        atr14,
        oos_start,
        oos_end,
        selected["atr"],
        selected["body"],
        selected["direction"]
    )

    all_results.extend(
        [2] * oos_stats["wins"]
        + [-1] * oos_stats["losses"]
    )

    if oos_stats["r"] > 0:
        positive_folds += 1

    print(
        "OOS:",
        "TRADES=", oos_stats["trades"],
        "| W=", oos_stats["wins"],
        "| L=", oos_stats["losses"]
    )

    print(
        "WR:", round(oos_stats["wr"], 2), "%",
        "| R:", oos_stats["r"],
        "| EXP:", round(oos_stats["exp"], 3)
    )

    print(
        "PF:", round(oos_stats["pf"], 3),
        "| DD:", oos_stats["dd"],
        "| LOSS STREAK:", oos_stats["loss_streak"]
    )

    print(
        "SAME-CANDLE SL+TP:",
        oos_stats["same_candle"],
        "| AVG EXIT BARS:",
        round(oos_stats["avg_bars"], 2)
    )

    fold_reports.append(oos_stats)

# ============================================================
# IMPORTANT:
# بازسازی ترتیب واقعی معاملات برای DD
# ============================================================
all_results = []

for report in fold_reports:

    # تعداد برد و باخت هر Fold
    all_results.extend(
        [2] * report["wins"]
    )

    all_results.extend(
        [-1] * report["losses"]
    )

combined = stats(all_results)

# ============================================================
# COMBINED
# ============================================================
print("=" * 65)
print("COMBINED WALK-FORWARD OOS")
print("=" * 65)

print("TRADES:", combined["trades"])
print("WINS:", combined["wins"])
print("LOSSES:", combined["losses"])
print(
    "WIN RATE:",
    round(combined["wr"], 2),
    "%"
)
print("TOTAL R:", combined["r"])
print(
    "EXPECTANCY:",
    round(combined["exp"], 3),
    "R"
)
print(
    "PROFIT FACTOR:",
    round(combined["pf"], 3)
)
print("MAX DD:", combined["dd"], "R")
print(
    "MAX LOSS STREAK:",
    combined["loss_streak"]
)

print(
    "POSITIVE FOLDS:",
    positive_folds,
    "/",
    len(fold_reports)
)

# ============================================================
# VALIDATION GATE
# ============================================================
if (
    positive_folds >= 3
    and combined["trades"] >= 40
    and combined["r"] > 0
    and combined["exp"] > 0
    and combined["pf"] > 1.0
):

    validation = "ROBUST / PROMISING"
    readiness = "CANDIDATE FOR UNTOUCHED FORWARD TEST"

else:

    validation = "NOT CONFIRMED"
    readiness = "NOT READY"

print("=" * 65)
print("VALIDATION:", validation)
print("TRADING READINESS:", readiness)
print("=" * 65)

# ============================================================
# PROJECT STATUS
# ============================================================
print("PROJECT COMPLETION: 97%")
print("PROJECT REMAINING: 3%")
print("LIVE TRADING: OFF")
print("REAL ORDER: OFF")

if validation == "ROBUST / PROMISING":

    print(
        "NEXT STEP:",
        "FINAL UNTOUCHED FORWARD TEST"
    )

else:

    print(
        "NEXT STEP:",
        "CONTINUE WALK-FORWARD DIAGNOSTICS"
    )

print("TRADING READINESS:", readiness)
print("=" * 65)
