from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from typer.testing import CliRunner

from llmbench.cli import app


class MockOpenAIHandler(BaseHTTPRequestHandler):
    def log_message(self, *_: object) -> None:
        return

    def do_GET(self) -> None:
        if self.path == "/v1/models":
            self._json({"data": [{"id": "mock-model"}]})
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", "0"))
        body = json.loads(self.rfile.read(length))
        prompt = body["messages"][-1]["content"]
        answer = '{"answer":"B"}' if "second" in prompt else "Paris"
        if body.get("stream"):
            chunks = [
                {"choices": [{"delta": {"content": answer}, "finish_reason": "stop"}]},
                {"choices": [], "usage": {"prompt_tokens": 8, "completion_tokens": 3}},
            ]
            payload = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
            payload += "data: [DONE]\n\n"
            encoded = payload.encode()
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("content-length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
        else:
            self._json(
                {
                    "choices": [{"message": {"content": answer}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 8, "completion_tokens": 3},
                }
            )

    def _json(self, payload: dict) -> None:
        encoded = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


@pytest.fixture
def mock_server() -> str:
    server = ThreadingHTTPServer(("127.0.0.1", 0), MockOpenAIHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_cli_catalog_commands() -> None:
    runner = CliRunner()
    datasets = runner.invoke(app, ["list-datasets"])
    assert datasets.exit_code == 0
    assert "gpqa-diamond" in datasets.stdout
    benchmarks = runner.invoke(app, ["list-benchmarks", "--bundled-only"])
    assert benchmarks.exit_code == 0
    assert "reports=225" in benchmarks.stdout


def test_cli_eval_end_to_end(mock_server: str, tmp_path) -> None:
    dataset = tmp_path / "custom.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "q1",
                "dataset": "custom",
                "type": "multiple_choice",
                "question": "Choose the second option.",
                "choices": {"A": "first", "B": "second"},
                "answer": "B",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "run"
    result = CliRunner().invoke(
        app,
        [
            "eval",
            "--base-url",
            mock_server,
            "--api-key",
            "EMPTY",
            "--dataset",
            str(dataset),
            "--limit",
            "1",
            "--concurrency",
            "1",
            "--output-dir",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.exception
    assert "Model: mock-model" in result.stdout
    assert "Requests: 1  Successful: 1  Score: 1.0000" in result.stdout
    assert f"Summary: {output / 'summary.json'}" in result.stdout
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["model"] == "mock-model"
    assert summary["schema_version"] == 2
    assert summary["quality"]["sample_mean_score"] == 1.0
    assert "pass_at_k" not in summary["quality"]
    assert summary["performance"]["successful_requests"] == 1
    assert summary["performance"]["truncation_rate"] == 0
    raw = json.loads((output / "raw_results.jsonl").read_text(encoding="utf-8"))
    assert raw["max_tokens"] == 4096
    assert (output / "events.jsonl").exists()
    assert (output / "run_manifest.json").exists()
    assert json.loads((output / "run_state.json").read_text())["status"] == "completed"

    resumed = CliRunner().invoke(
        app,
        ["eval", "--api-key", "EMPTY", "--resume", str(output)],
    )
    assert resumed.exit_code == 0, resumed.exception
    assert len((output / "raw_results.jsonl").read_text().splitlines()) == 1


def test_cli_streaming_stress(mock_server: str, tmp_path) -> None:
    output = tmp_path / "stress"
    result = CliRunner().invoke(
        app,
        [
            "stress",
            "--base-url",
            mock_server,
            "--api-key",
            "EMPTY",
            "--concurrency",
            "2",
            "--duration",
            "0",
            "--requests",
            "4",
            "--output-dir",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.exception
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["performance"]["total_requests"] == 4
    assert summary["performance"]["ttft_ms"]["p95"] is not None
