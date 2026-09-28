# FarmandehAI New Bot

GBP/USD 5-minute trading research project based on the supplied specification.

## Safety
- PAPER=True
- LIVE=False
- REAL=False
- CLOSED_ONLY=True
- NO_LOOKAHEAD=True
- No real order execution exists in this version.

## Run tests
```bash
pip install -r requirements.txt
pytest -q
```

## Backtest with CSV
CSV columns: `timestamp,open,high,low,close` and optional `volume`.

```bash
python main.py --csv path/to/gbpusd_5m.csv
```

## TwelveData
Set `TWELVEDATA_API_KEY` in the environment. Never put it in source code or reports.

```bash
export TWELVEDATA_API_KEY='...'
python main.py
```

Validation remains `NOT CONFIRMED` until genuine data and the required tests are completed.
