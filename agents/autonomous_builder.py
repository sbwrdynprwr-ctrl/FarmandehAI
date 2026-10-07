from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "artifacts" / "autonomous_builder_state.json"
MODEL = os.getenv("BUILDER_MODEL", "gpt-6-luna")
MAX_REPAIR_ATTEMPTS = int(os.getenv("BUILDER_MAX_REPAIR_ATTEMPTS", "3"))

SYSTEM = """
You are the autonomous software-engineering agent for FarmandehAI.
Improve the existing repository incrementally. Do not redesign it from scratch.

Hard safety rules:
- This repository is paper-only. Never enable LIVE or REAL trading.
- Never add broker/exchange order submission or secret logging.
- Never put API keys or secrets in source code.
- Preserve no-lookahead, closed-only and existing safety gates.
- Prefer small, testable changes.
- If no safe improvement is justified, return an empty diff.

Return ONLY a unified git diff. No markdown fences.
"""

def sh(*args: str, check: bool = True) -> str:
    p = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=False)
    if check and p.returncode:
        raise RuntimeError(p.stderr or p.stdout or "command failed")
    return p.stdout

def snapshot() -> str:
    paths = sh("git", "ls-files").splitlines()
    chunks = []
    for path in paths:
        if path.startswith(".git/") or path.startswith("artifacts/"):
            continue
        if not path.endswith((".py", ".md", ".yml", ".yaml", ".json", ".txt")):
            continue
        try:
            data = (ROOT / path).read_text(encoding="utf-8")
        except Exception:
            continue
        if len(data) > 10000:
            data = data[:10000] + "\\n...[truncated]"
        chunks.append("\\n===== " + path + " =====\\n" + data)
    return "".join(chunks[:80])

def ask_model(repo: str, failure: str | None = None) -> str:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    prompt = (
        "Inspect this repository and choose ONE highest-value safe improvement "
        "that can be completed in this run. Add tests when appropriate.\n\n" + repo
    )
    if failure:
        prompt += "\n\nThe previous attempt failed validation. Diagnose and fix that failure while keeping the change minimal:\n" + failure[-12000:]
    response = client.responses.create(model=MODEL, instructions=SYSTEM, input=prompt)
    return response.output_text.strip()

def safety_scan(diff: str) -> None:
    changed = [line[6:] for line in diff.splitlines() if line.startswith("+++ b/")]
    for path in changed:
        if path.startswith(".github/workflows/") or path == ".gitignore":
            raise RuntimeError("safety scan rejected protected path: " + path)
    forbidden = [
        "LIVE=true", "LIVE = True", "REAL=true", "REAL = True",
        "ccxt.create_order", "create_market_order", "create_limit_order",
        "withdraw(", "send_transaction(", "private_key",
    ]
    low = diff.lower()
    for token in forbidden:
        if token.lower() in low:
            raise RuntimeError("safety scan rejected generated diff: " + token)

def run_tests() -> None:
    sh("python", "-m", "compileall", "-q", ".")
    sh("python", "-m", "pytest", "-q")
    sh("git", "diff", "--check")

def main() -> int:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    before = sh("git", "rev-parse", "HEAD").strip()
    failure = None
    diff = ""
    for attempt in range(1, MAX_REPAIR_ATTEMPTS + 1):
        diff = ask_model(snapshot(), failure)
        safety_scan(diff)
        if not diff:
            break
        fd, patch = tempfile.mkstemp(suffix=".patch")
        os.close(fd)
        try:
            Path(patch).write_text(diff, encoding="utf-8")
            sh("git", "apply", "--check", patch)
            sh("git", "apply", patch)
            run_tests()
            break
        except Exception as exc:
            failure = str(exc)
            subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=ROOT, check=False)
            if attempt == MAX_REPAIR_ATTEMPTS:
                raise RuntimeError(f"validation failed after {MAX_REPAIR_ATTEMPTS} attempts: {failure}")
        finally:
            try:
                os.unlink(patch)
            except OSError:
                pass

    if not diff:
        STATE.write_text(json.dumps({
            "status": "NO_CHANGE", "base": before, "model": MODEL
        }, indent=2), encoding="utf-8")
        return 0

    sh("git", "add", "-A")
    sh("git", "commit", "-m", "chore(agent): autonomous safe improvement")
    after = sh("git", "rev-parse", "HEAD").strip()

    STATE.write_text(json.dumps({
        "status": "CHANGED_AND_TESTED",
        "base": before,
        "commit": after,
        "model": MODEL,
        "repair_attempts": MAX_REPAIR_ATTEMPTS,
    }, indent=2), encoding="utf-8")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
