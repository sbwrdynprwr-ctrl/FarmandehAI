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
The repository contains the paper-only orchestrator, persisted validation artifact, safety gate, checkpoint-resume support, and automated tests.

## Deployment chain
The canonical deployment source is `main`. Every push to `main` is covered by GitHub Actions CI, and the Railway service is connected to `main`. A strategy is not considered approved merely because CI passes: independent holdout, walk-forward robustness, spread sensitivity, and Monte Carlo gates must also pass.

## Research gate
No live/real execution is permitted by the automation chain. The next promotion step after a passing statistical gate is paper-only shadow validation; external AI critics are added only after the statistical baseline is robust.


Research validation checkpoint updated: worst-slice neighbor stability screening remains enforced.


Research validation checkpoint: 2026-10-05T06:35:23.805Z


<!-- research-trigger: v2 activity recovery -->

<!-- research trigger: v2 target-floor expansion 2026-10-05 -->