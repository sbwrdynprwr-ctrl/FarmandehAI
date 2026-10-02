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

## TwelveData
Set `TWELVEDATA_API_KEY` in the environment. Never put it in source code or reports.

```bash
export TWELVEDATA_API_KEY='...'
python main.py
```

Validation remains `NOT CONFIRMED` until genuine data and the required tests are completed.

## Research runtime
The research runtime uses the strategy module committed in the repository. It does not download or overwrite source code at startup, so test execution remains isolated from network availability.

## Holdout runtime marker
The current holdout runner is intentionally executed with real/live trading disabled while the independent forward validation is completed.

## Automation checkpoint
The repository now contains the paper-only orchestrator, persisted validation artifact, safety gate, and automated tests. Railway deployment must run the current `main` commit before this checkpoint is considered active.

## Railway automation checkpoint
The latest `main` commit is the intended deployment target; live/real execution remains disabled.
