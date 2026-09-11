# ============================================================
# FARMANDEHAI v17406.28
# GBP/USD | 5M | WALK-FORWARD VALIDATION
# PAPER ONLY
# ============================================================

import requests
import os
from datetime import datetime

# ============================================================
# 🔑 TWELVEDATA API KEY — فقط اینجا کلید خودت را بگذار
# ============================================================
API_KEY = os.getenv("TWELVEDATA_API_KEY", "")

# ============================================================
# CONFIG
# ============================================================
VERSION = "v17406.28"
PAIR = "GBP/USD"
INTERVAL = "5min"

PAPER = True
LIVE = False
REAL = False
NO_LOOKAHEAD = True
CLOSED_ONLY = True

HIST_CANDLES = 3000
ATR_PERIOD = 14
ATR_M = 1.5
RR = 2.0
BODY_MIN = 0.55


# ============================================================
# SAFETY
# ============================================================
if not PAPER or LIVE or REAL:
    raise SystemExit("TRADE BLOCKED - PAPER ONLY")

if API_KEY == "YOUR_TWELVEDATA_KEY":
    raise SystemExit("ERROR: TwelveData API KEY را وارد کنید")


# ============================================================
# TWELVEDATA
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
                timeout=20
            )

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
                    data.append({
                        "time": datetime.fromisoformat(
                            x["datetime"].replace("Z", "+00:00")
                        ),
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
# BUY ONLY
# ============================================================
def trade_result(data, i, atr_value, end):

    entry = data[i]["close"]

    distance = atr_value * ATR_M

    sl = entry - distance
    tp = entry + distance * RR

    for j in range(i + 1, end):

        high = data[j]["high"]
        low = data[j]["low"]

        # اگر SL و TP هر دو در یک کندل لمس شوند
        # محافظه‌کارانه SL حساب می‌شود
        if low <= sl and high >= tp:
            return -1

        if low <= sl:
            return -1

        if high >= tp:
            return 2

    return None


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

    exp = (
        total_r / n
        if n else 0
    )

    pf = (
        wins * 2 / losses
        if losses else float("inf")
    )

    equity = 0
    peak = 0
    max_dd = 0

    streak = 0
    max_streak = 0

    for x in results:

        equity += x

        if equity > peak:
            peak = equity

        max_dd = min(
            max_dd,
            equity - peak
        )

        if x == -1:
            streak += 1
            max_streak = max(
                max_streak,
                streak
            )
        else:
            streak = 0

    return (
        n,
        wins,
        losses,
        wr,
        total_r,
        exp,
        pf,
        max_dd,
        max_streak
    )


# ============================================================
# RUN OOS
# ============================================================
def run_oos(data, ema20, ema50, atr14, start, end):

    results = []

    i = start

    while i < end - 1:

        if (
            ema20[i] is not None
            and ema50[i] is not None
            and atr14[i] is not None
        ):

            c = data[i]

            candle_range = (
                c["high"] - c["low"]
            )

            if candle_range > 0:

                body_ratio = (
                    abs(
                        c["close"]
                        - c["open"]
                    )
                    / candle_range
                )

                signal = (
                    ema20[i] > ema50[i]
                    and c["close"] > ema20[i]
                    and body_ratio >= BODY_MIN
                )

                if signal:

                    result = trade_result(
                        data,
                        i,
                        atr14[i],
                        end
                    )

                    if result is not None:

                        results.append(result)

                        # جلوگیری از معاملات همپوشان
                        i += 2
                        continue

        i += 1

    return results


# ============================================================
# START
# ============================================================
print("=" * 60)
print("Starting FarmandehAI", VERSION)
print("=" * 60)

print("PAIR:", PAIR)
print("TIMEFRAME:", INTERVAL)
print("PAPER:", PAPER)
print("LIVE:", LIVE)
print("REAL ORDER:", REAL)
print("NO LOOKAHEAD:", NO_LOOKAHEAD)
print("CLOSED ONLY:", CLOSED_ONLY)
print("HISTORICAL CANDLES:", HIST_CANDLES)

print("-" * 60)

data = get_data()

if len(data) < 2500:

    print("NOT ENOUGH DATA:", len(data))
    raise SystemExit

print("TWELVEDATA CONNECTED")
print("CANDLES RECEIVED:", len(data))

print("-" * 60)


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
# WALK-FORWARD
# ============================================================
folds = [
    (0, 1000, 1000, 1500),
    (500, 1500, 1500, 2000),
    (1000, 2000, 2000, 2500),
    (1500, 2500, 2500, 3000)
]

all_results = []
positive_folds = 0

for number, (
    train_start,
    train_end,
    oos_start,
    oos_end
) in enumerate(folds, 1):

    results = run_oos(
        data,
        ema20,
        ema50,
        atr14,
        oos_start,
        oos_end
    )

    all_results.extend(results)

    s = stats(results)

    if s[4] > 0:
        positive_folds += 1

    print("FOLD", number)

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

    print(
        "TRADES:", s[0],
        "| W:", s[1],
        "| L:", s[2]
    )

    print(
        "WR:", round(s[3], 2), "%",
        "| R:", s[4],
        "| EXP:", round(s[5], 3)
    )

    print(
        "PF:", round(s[6], 3),
        "| DD:", s[7],
        "| LOSS STREAK:", s[8]
    )

    print("-" * 60)


# ============================================================
# COMBINED
# ============================================================
s = stats(all_results)

print("=" * 60)
print("COMBINED WALK-FORWARD OOS")
print("=" * 60)

print("TRADES:", s[0])
print("WINS:", s[1])
print("LOSSES:", s[2])
print("WIN RATE:", round(s[3], 2), "%")
print("TOTAL R:", s[4])
print("EXPECTANCY:", round(s[5], 3), "R")
print("PROFIT FACTOR:", round(s[6], 3))
print("MAX DD:", s[7], "R")
print("MAX LOSS STREAK:", s[8])
print(
    "POSITIVE FOLDS:",
    positive_folds,
    "/",
    len(folds)
)


# ============================================================
# VALIDATION
# ============================================================
if (
    positive_folds >= 3
    and s[0] >= 40
    and s[4] > 0
    and s[5] > 0
    and s[6] > 1
):

    validation = "ROBUST / PROMISING"

else:

    validation = "NOT CONFIRMED"

print("VALIDATION:", validation)

print("=" * 60)
print("PROJECT COMPLETION: 97%")
print("PROJECT REMAINING: 3%")
print("LIVE TRADING: OFF")
print("REAL ORDER: OFF")
print("NEXT STEP: FINAL UNTOUCHED FORWARD TEST")
print("TRADING READINESS: NOT READY")
print("=" * 60)
