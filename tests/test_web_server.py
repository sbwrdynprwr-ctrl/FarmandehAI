import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import urlopen

from web_server import load_readonly_status, make_handler


def test_readonly_status_fails_safe_when_artifact_is_missing(tmp_path):
    status = load_readonly_status(tmp_path)
    assert status["mode"] == "PAPER"
    assert status["paper_trading"] is True
    assert status["live_trading"] is False
    assert status["real_orders"] is False
    assert status["validation"] == "NOT CONFIRMED"
    assert status["automation_overall"] == "UNKNOWN"


def test_readonly_status_reads_non_secret_project_markers(tmp_path):
    (tmp_path / "project_status.txt").write_text(
        "PROJECT COMPLETION: 92%\n"
        "PROJECT REMAINING: 8%\n"
        "VERSION: test-version\n"
        "VALIDATION: NOT CONFIRMED\n"
        "NEXT STEP: Keep testing.\n",
        encoding="utf-8",
    )
    status = load_readonly_status(tmp_path)
    assert status["project_completion_recorded"] == "92%"
    assert status["project_remaining_recorded"] == "8%"
    assert status["version"] == "test-version"
    assert status["next_step"] == "Keep testing."
    assert status["real_orders"] is False


def test_automation_status_exposes_only_safe_stage_fields(tmp_path):
    (tmp_path / "automation_state.json").write_text(
        json.dumps({
            "overall": "RUNNING",
            "updated_at": "2026-10-08T12:00:00Z",
            "stages": [
                {"name": "tests", "status": "SUCCESS", "detail": "private detail"},
                {"name": "validation", "status": "RUNNING", "detail": "private detail"},
            ],
        }),
        encoding="utf-8",
    )
    status = load_readonly_status(tmp_path)
    assert status["automation_overall"] == "RUNNING"
    assert status["automation_stages"] == [
        {"name": "tests", "status": "SUCCESS"},
        {"name": "validation", "status": "RUNNING"},
    ]
    assert "detail" not in json.dumps(status)


def test_http_status_endpoint_is_read_only_and_serves_ui(tmp_path):
    web_dir = tmp_path / "web"
    artifacts_dir = tmp_path / "artifacts"
    web_dir.mkdir()
    artifacts_dir.mkdir()
    (web_dir / "index.html").write_text("<h1>Command Center</h1>", encoding="utf-8")
    (artifacts_dir / "project_status.txt").write_text(
        "VALIDATION: NOT CONFIRMED\n", encoding="utf-8"
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(web_dir, artifacts_dir))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        with urlopen(base + "/api/status", timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
            assert response.status == 200
            assert payload["paper_trading"] is True
            assert payload["live_trading"] is False
            assert payload["real_orders"] is False
        with urlopen(base + "/", timeout=3) as response:
            assert "Command Center" in response.read().decode("utf-8")
    finally:
        server.shutdown()
        server.server_close()
