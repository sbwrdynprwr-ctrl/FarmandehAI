from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

ARTIFACTS = Path("artifacts")
STATE_FILE = ARTIFACTS / "automation_state.json"


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def check_safety() -> str:
    """Hard paper-trading safety gate used by the automation chain."""
    if _bool("LIVE", False) or _bool("REAL", False):
        raise RuntimeError("Safety lock: LIVE/REAL must remain disabled")
    if not _bool("PAPER", True):
        raise RuntimeError("Safety lock: PAPER must remain enabled")
    return "PAPER=True LIVE=False REAL=False"


@dataclass
class StageResult:
    name: str
    status: str
    attempts: int
    started_at: str
    finished_at: str
    detail: str = ""


class AutomationChain:
    """Deterministic paper-only chain with bounded retries and persisted state.

    Downstream stages never run after a failed prerequisite. LIVE/REAL are
    always hard-disabled; this orchestrator does not contain an order-routing
    path.
    """

    def __init__(self, retries: int = 1, retry_delay: float = 5.0):
        self.retries = max(0, int(retries))
        self.retry_delay = max(0.0, float(retry_delay))
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        self.results: list[StageResult] = []

    def _write_state(self, overall: str = "RUNNING") -> None:
        payload = {
            "overall": overall,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "paper": True,
            "live": False,
            "real": False,
            "stages": [asdict(x) for x in self.results],
        }
        STATE_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def run_stage(self, name: str, fn: Callable[[], str]) -> StageResult:
        started = datetime.now(timezone.utc).isoformat()
        last_error = ""
        for attempt in range(1, self.retries + 2):
            print(f"CHAIN_STAGE_START={name} ATTEMPT={attempt}", flush=True)
            try:
                detail = fn() or "ok"
                result = StageResult(
                    name=name, status="SUCCESS", attempts=attempt,
                    started_at=started,
                    finished_at=datetime.now(timezone.utc).isoformat(),
                    detail=detail,
                )
                self.results.append(result)
                self._write_state()
                print(f"CHAIN_STAGE_DONE={name} STATUS=SUCCESS ATTEMPTS={attempt}", flush=True)
                return result
            except BaseException as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                print(f"CHAIN_STAGE_ERROR={name} ATTEMPT={attempt} ERROR={last_error}", flush=True)
                if attempt <= self.retries:
                    time.sleep(self.retry_delay)
        result = StageResult(
            name=name, status="FAILED", attempts=self.retries + 1,
            started_at=started,
            finished_at=datetime.now(timezone.utc).isoformat(),
            detail=last_error,
        )
        self.results.append(result)
        self._write_state("FAILED")
        print(f"CHAIN_STAGE_DONE={name} STATUS=FAILED", flush=True)
        raise RuntimeError(f"stage {name} failed: {last_error}")

    def run(self) -> int:
        self._write_state()

        def tests():
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "-q"],
                check=False,
                text=True,
            )
            if proc.returncode:
                raise RuntimeError(f"pytest exited with {proc.returncode}")
            return "pytest passed"

        def validation():
            proc = subprocess.run(
                [sys.executable, "-u", "holdout_runner.py"],
                check=False,
                text=True,
            )
            if proc.returncode:
                raise RuntimeError(f"holdout_runner exited with {proc.returncode}")
            return "independent holdout + robustness + Monte Carlo completed"

        def paper_gate():
            # The project currently has no broker/exchange execution path.
            # This stage records readiness only and never submits a live order.
            validation_state = self._read_validation_marker()
            if validation_state is None:
                raise RuntimeError("validation marker missing")
            return f"paper-only gate recorded: approved={validation_state.get('approved', [])}"

        self.run_stage("safety", check_safety)
        self.run_stage("tests", tests)
        self.run_stage("validation", validation)
        self.run_stage("paper_gate", paper_gate)
        self._write_state("SUCCESS")
        Path(ARTIFACTS / "automation_complete.txt").write_text(
            "AUTOMATION_CHAIN=SUCCESS\nLIVE=False\nREAL=False\nPAPER=True\n",
            encoding="utf-8",
        )
        print("FARMANDEHAI_AUTOMATION_DONE", flush=True)
        return 0

    @staticmethod
    def _read_validation_marker():
        marker = Path(ARTIFACTS / "validation.json")
        if marker.exists():
            try:
                return json.loads(marker.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        # holdout_runner writes its result to stdout; requiring the artifact
        # keeps the chain machine-readable instead of scraping logs.
        raise RuntimeError("artifacts/validation.json was not produced")


if __name__ == "__main__":
    print("FARMANDEHAI_AUTOMATION_START", flush=True)
    try:
        AutomationChain(
            retries=int(os.getenv("CHAIN_RETRIES", "1")),
            retry_delay=float(os.getenv("CHAIN_RETRY_DELAY_SECONDS", "5")),
        ).run()
    except BaseException as exc:
        print(f"FARMANDEHAI_AUTOMATION_ERROR={type(exc).__name__}:{exc}", flush=True)
        traceback.print_exc()
        raise
