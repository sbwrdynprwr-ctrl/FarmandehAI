import json
from pathlib import Path

import pytest

from orchestrator import AutomationChain, check_safety


def test_stage_chain_stops_after_failure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    chain = AutomationChain(retries=1, retry_delay=0)
    seen = []

    def ok():
        seen.append("ok")
        return "ok"

    def fail():
        seen.append("fail")
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        chain.run_stage("first", ok)
        chain.run_stage("second", fail)
        chain.run_stage("third", ok)

    assert seen == ["ok", "fail", "fail"]
    state = json.loads(Path("artifacts/automation_state.json").read_text())
    assert state["overall"] == "FAILED"
    assert state["stages"][-1]["name"] == "second"
    assert state["stages"][-1]["attempts"] == 2


def test_safety_rejects_live(monkeypatch):
    monkeypatch.setenv("LIVE", "true")
    monkeypatch.setenv("REAL", "false")
    chain = AutomationChain(retries=0)
    with pytest.raises(RuntimeError):
        chain.run_stage("safety", lambda: (_ for _ in ()).throw(RuntimeError("Safety lock: LIVE/REAL must remain disabled")))


def test_validation_marker_is_required(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    chain = AutomationChain(retries=0)
    Path("artifacts").mkdir()
    with pytest.raises(RuntimeError, match="validation marker missing"):
        chain.run_stage("paper_gate", lambda: f"paper-only gate recorded: approved={chain._read_validation_marker().get('approved', [])}")
