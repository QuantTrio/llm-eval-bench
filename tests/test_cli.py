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
    validated = CliRunner().invoke(app, ["validate", str(output)])
    assert validated.exit_code == 0, validated.exception
    assert "summary.json" in validated.stdout

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


def test_cli_yaml_config_and_cli_override(mock_server: str, tmp_path, monkeypatch) -> None:
    dataset = tmp_path / "custom.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "q1",
                "dataset": "custom",
                "type": "exact_match",
                "question": "Capital of France?",
                "answer": "Paris",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "configured-run"
    config = tmp_path / "bench.yaml"
    config.write_text(
        "\n".join(
            [
                "schema_version: 2",
                "targets:",
                "  chat:",
                f"    base_url: {mock_server}",
                "    model: mock-model",
                "    api_key_env: TEST_BENCH_KEY",
                "run:",
                f"  datasets: ['{dataset}']",
                "  limit_per_dataset: 1",
                "  max_tokens: 16",
                f"  output_dir: {output}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TEST_BENCH_KEY", "EMPTY")
    result = CliRunner().invoke(
        app,
        ["eval", "--config", str(config), "--max-tokens", "32"],
    )
    assert result.exit_code == 0, result.exception
    raw = json.loads((output / "raw_results.jsonl").read_text())
    assert raw["max_tokens"] == 32


def test_cli_concurrency_sweep(mock_server: str, tmp_path) -> None:
    output = tmp_path / "sweep"
    result = CliRunner().invoke(
        app,
        [
            "sweep",
            "--base-url",
            mock_server,
            "--api-key",
            "EMPTY",
            "--model",
            "mock-model",
            "--concurrency",
            "1,2",
            "--warmup-requests",
            "1",
            "--requests",
            "2",
            "--prompt-profile",
            "short",
            "--no-server-metrics",
            "--output-dir",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.exception
    payload = json.loads((output / "sweep.json").read_text())
    assert [point["concurrency"] for point in payload["points"]] == [1, 2]
    assert all(
        point["summary"]["performance"]["total_requests"] == 2 for point in payload["points"]
    )
    sweep_html = (output / "sweep.html").read_text(encoding="utf-8")
    assert "QPS by concurrency" in sweep_html
    assert "p95 latency (ms) by concurrency" in sweep_html


def test_cli_capability_suite_native_chat(mock_server: str, tmp_path, monkeypatch) -> None:
    dataset = tmp_path / "suite.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "q1",
                "dataset": "suite-custom",
                "type": "exact_match",
                "question": "Capital of France?",
                "answer": "Paris",
                "metadata": {
                    "benchmark_category": "Test",
                    "benchmark_metric": "exact_match",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "suite-run"
    config = tmp_path / "suite.yaml"
    config.write_text(
        "\n".join(
            [
                "schema_version: 2",
                "targets:",
                "  chat:",
                f"    base_url: {mock_server}",
                "    model: mock-model",
                "    api_key_env: SUITE_CHAT_KEY",
                "run:",
                "  concurrency: 1",
                "  stream: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SUITE_CHAT_KEY", "EMPTY")
    result = CliRunner().invoke(
        app,
        [
            "suite",
            "--config",
            str(config),
            "--dataset",
            str(dataset),
            "--limit",
            "1",
            "--output-dir",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.exception
    summary = json.loads((output / "summary.json").read_text())
    assert summary["coverage"]["ratio"] == 1
    assert summary["quality"]["by_dataset"]["suite-custom"]["score"] == 1
