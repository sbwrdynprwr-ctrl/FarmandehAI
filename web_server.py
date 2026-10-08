from __future__ import annotations

import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
ARTIFACTS_DIR = ROOT / "artifacts"


def _read_status_file(artifacts_dir: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    status_path = artifacts_dir / "project_status.txt"
    try:
        lines = status_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for line in lines:
        if ": " in line:
            key, value = line.split(": ", 1)
            values[key.strip().upper()] = value.strip()
    return values


def load_readonly_status(artifacts_dir: Path = ARTIFACTS_DIR) -> dict[str, Any]:
    """Expose only non-secret, read-only project status; never expose order controls."""
    values = _read_status_file(artifacts_dir)
    return {
        "application": "FarmandehAI",
        "mode": "PAPER",
        "paper_trading": True,
        "live_trading": False,
        "real_orders": False,
        "validation": values.get("VALIDATION", "NOT CONFIRMED"),
        "project_completion_recorded": values.get("PROJECT COMPLETION", "unknown"),
        "project_remaining_recorded": values.get("PROJECT REMAINING", "unknown"),
        "version": values.get("VERSION", "unknown"),
        "next_step": values.get("NEXT STEP", "Review validation evidence; keep live trading disabled."),
        "data_note": "This endpoint exposes project status only; it does not provide live market data or execute trades.",
    }


def make_handler(web_dir: Path = WEB_DIR, artifacts_dir: Path = ARTIFACTS_DIR):
    web_root = web_dir.resolve()
    artifacts_root = artifacts_dir.resolve()

    class FarmandehAIHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(web_root), **kwargs)

        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path.split("?", 1)[0] == "/api/health":
                self._send_json({"ok": True, "service": "FarmandehAI read-only UI"})
                return
            if self.path.split("?", 1)[0] == "/api/status":
                self._send_json(load_readonly_status(artifacts_root))
                return
            if self.path.split("?", 1)[0].startswith("/api/"):
                self._send_json({"error": "read-only endpoint not found"}, 404)
                return
            super().do_GET()

        # Deliberately no POST/PUT/PATCH/DELETE implementation: this is a read-only UI server.

    return FarmandehAIHandler


def main() -> None:
    port = int(os.environ.get("PORT", "8080"))
    server = ThreadingHTTPServer(("0.0.0.0", port), make_handler())
    print(f"FARMANDEHAI_READONLY_UI_PORT={port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
