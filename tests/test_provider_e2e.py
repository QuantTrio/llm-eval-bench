"""Loopback HTTP acceptance using realistic provider wire formats and explicit model IDs."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
from typer.testing import CliRunner

from llmbench.cli import app

APIS = ("chat", "responses", "messages", "generate-content")
MODEL_ID = "fixture-model"
ANSWER = '{"answer":"72"}'
ARTIFACTS = (
    "summary.json",
    "raw_results.jsonl",
    "run_manifest.json",
    "events.jsonl",
    "run_state.json",
    "report.html",
)


def make_dataset(path: Path, *, total: int = 2) -> None:
    path.write_text(
        "".join(
            json.dumps(
                {
                    "id": f"q-{index}",
                    "dataset": "local-math",
                    "type": "math",
                    "question": "What is 6 * 12?",
                    "answer": "72",
                }
            )
            + "\n"
            for index in range(total)
        ),
        encoding="utf-8",
    )


class MockState:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.force_status: int | None = None
        self.answer = ANSWER
        self.finish_reason = "stop"
        self.reasoning = False


def _sse(payload: dict, *, event: str | None = None) -> str:
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {json.dumps(payload)}\n\n"


def _responses_payload(state: MockState) -> dict:
    output = []
    if state.reasoning:
        output.append(
            {
                "type": "reasoning",
                "id": "rs_fixture",
                "summary": [{"type": "summary_text", "text": '{"answer":"0"}'}],
            }
        )
    output.append(
        {
            "type": "message",
            "id": "msg_fixture",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": state.answer, "annotations": []}],
        }
    )
    return {
        "id": "resp_fixture",
        "object": "response",
        "status": "completed",
        "model": MODEL_ID,
        "output": output,
        "usage": {
            "input_tokens": 6,
            "output_tokens": 5,
            "total_tokens": 11,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


def _response(api: str, state: MockState, stream: bool) -> dict | str:
    if api == "chat":
        usage = {"prompt_tokens": 6, "completion_tokens": 5, "total_tokens": 11}
        if not stream:
            return {
                "id": "chatcmpl_fixture",
                "model": MODEL_ID,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": state.answer,
                        },
                        "finish_reason": state.finish_reason,
                    }
                ],
                "usage": usage,
            }
        return (
            _sse({"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
            + _sse(
                {
                    "choices": [
                        {"index": 0, "delta": {"content": state.answer}, "finish_reason": None}
                    ]
                }
            )
            + _sse({"choices": [{"index": 0, "delta": {}, "finish_reason": state.finish_reason}]})
            + _sse({"choices": [], "usage": usage})
            + "data: [DONE]\n\n"
        )
    if api == "responses":
        payload = _responses_payload(state)
        if not stream:
            return payload
        return (
            _sse(
                {
                    "type": "response.created",
                    "response": {
                        "id": "resp_fixture",
                        "status": "in_progress",
                        "output": [],
                    },
                },
                event="response.created",
            )
            + _sse(
                {
                    "type": "response.output_text.delta",
                    "item_id": "msg_fixture",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": state.answer,
                },
                event="response.output_text.delta",
            )
            + _sse(
                {
                    "type": "response.output_text.done",
                    "item_id": "msg_fixture",
                    "output_index": 0,
                    "content_index": 0,
                    "text": state.answer,
                },
                event="response.output_text.done",
            )
            + _sse({"type": "response.completed", "response": payload}, event="response.completed")
        )
    if api == "messages":
        message = {
            "id": "msg_fixture",
            "type": "message",
            "role": "assistant",
            "model": MODEL_ID,
            "content": [{"type": "text", "text": state.answer}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 6, "output_tokens": 5},
        }
        if not stream:
            return message
        return (
            _sse(
                {
                    "type": "message_start",
                    "message": {
                        **message,
                        "content": [],
                        "stop_reason": None,
                        "usage": {"input_tokens": 6, "output_tokens": 0},
                    },
                },
                event="message_start",
            )
            + _sse(
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
                event="content_block_start",
            )
            + _sse(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": state.answer},
                },
                event="content_block_delta",
            )
            + _sse({"type": "content_block_stop", "index": 0}, event="content_block_stop")
            + _sse(
                {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": "end_turn",
                        "stop_sequence": None,
                    },
                    "usage": {"output_tokens": 5},
                },
                event="message_delta",
            )
            + _sse({"type": "message_stop"}, event="message_stop")
        )
    payload = {
        "candidates": [
            {
                "index": 0,
                "content": {
                    "role": "model",
                    "parts": [{"text": state.answer}],
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 6,
            "candidatesTokenCount": 5,
            "totalTokenCount": 11,
            "thoughtsTokenCount": 0,
        },
        "modelVersion": MODEL_ID,
        "responseId": "gemini_fixture",
    }
    return _sse(payload) if stream else payload


@contextmanager
def run_local_server(state: MockState) -> Iterator[str]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: object) -> None:
            pass

        def _record(self) -> dict:
            parsed = urlparse(self.path)
            length = int(self.headers.get("content-length", "0"))
            call = {
                "method": self.command,
                "path": parsed.path,
                "query": parse_qs(parsed.query),
                "body": json.loads(self.rfile.read(length)) if length else {},
                "headers": {key.lower(): value for key, value in self.headers.items()},
            }
            state.calls.append(call)
            return call

        def _write(self, body: dict | str, status: int = 200) -> None:
            encoded = (body if isinstance(body, str) else json.dumps(body)).encode()
            self.send_response(status)
            self.send_header(
                "content-type", "text/event-stream" if isinstance(body, str) else "application/json"
            )
            self.send_header("content-length", str(len(encoded)))
            self.send_header("x-request-id", "fixture-request")
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:
            self._record()
            self._write({"data": [{"id": "wrong-first-model"}, {"id": MODEL_ID}]})

        def do_POST(self) -> None:
            call = self._record()
            if state.force_status is not None:
                self._write({"error": {"message": "fixture failure"}}, state.force_status)
                return
            path, body = call["path"], call["body"]
            if path.endswith("/chat/completions"):
                api = "chat"
            elif path.endswith("/responses"):
                api = "responses"
            elif path.endswith("/messages"):
                api = "messages"
            elif path.endswith((":generateContent", ":streamGenerateContent")):
                api = "generate-content"
            else:
                self._write({"error": "unknown path"}, 404)
                return
            if api != "generate-content" and body.get("model") != MODEL_ID:
                self._write({"error": "explicit fixture model required"}, 400)
                return
            stream = body.get("stream", False) or path.endswith(":streamGenerateContent")
            self._write(_response(api, state, stream))

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.fixture(autouse=True)
def no_hosted_keys(monkeypatch) -> None:
    for key in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "XAI_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "CHAT_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)


def invoke_run(
    *,
    base_url: str,
    api: str,
    dataset: Path,
    output_dir: Path,
    stream: bool = False,
    extra: list[str] | None = None,
):
    command = [
        "run",
        "--base-url",
        base_url,
        "--api",
        api,
        "--model",
        MODEL_ID,
        "--dataset",
        str(dataset),
        "--limit",
        "2",
        "--max-tokens",
        "64",
        "--concurrency",
        "2",
        "--mode",
        "quality",
        "--retries",
        "0",
        "--output-dir",
        str(output_dir),
    ]
    if not stream:
        command.append("--no-stream")
    return CliRunner().invoke(app, [*command, *(extra or [])])


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize("stream", [False, True], ids=["json", "sse"])
def test_provider_endpoint_to_scored_artifacts(api: str, stream: bool, tmp_path: Path) -> None:
    state = MockState()
    dataset, output = tmp_path / "custom.jsonl", tmp_path / "result"
    make_dataset(dataset)
    version = "v1beta" if api == "generate-content" else "v1"
    with run_local_server(state) as base:
        result = invoke_run(
            base_url=f"{base}/proxy/{version}",
            api=api,
            dataset=dataset,
            output_dir=output,
            stream=stream,
        )
    assert result.exit_code == 0, (result.output, result.exception)
    assert len(state.calls) == 2
    suffix = {
        "chat": "/chat/completions",
        "responses": "/responses",
        "messages": "/messages",
        "generate-content": f"/models/{MODEL_ID}:"
        + ("streamGenerateContent" if stream else "generateContent"),
    }[api]
    for call in state.calls:
        assert call["method"] == "POST"  # Never select the first GET /models entry.
        assert call["path"] == f"/proxy/{version}{suffix}"
        assert call["query"] == ({"alt": ["sse"]} if api == "generate-content" and stream else {})
        assert not any(
            name in call["headers"] for name in ("authorization", "x-api-key", "x-goog-api-key")
        )
        if api == "generate-content":
            assert "contents" in call["body"]
            assert call["body"]["generationConfig"]["maxOutputTokens"] == 64
        else:
            assert call["body"]["model"] == MODEL_ID
            assert ("input" if api == "responses" else "messages") in call["body"]

    for name in ARTIFACTS:
        assert (output / name).is_file(), name
    summary = _read_json(output / "summary.json")
    manifest = _read_json(output / "run_manifest.json")
    rows = _read_rows(output / "raw_results.jsonl")
    events = _read_rows(output / "events.jsonl")
    assert summary["model"] == manifest["model"] == MODEL_ID
    assert summary["config"]["api"] == manifest["config"]["api"] == api
    assert summary["quality"]["sample_mean_score"] == 1.0
    assert summary["quality"]["quality_valid"] is True
    assert summary["performance"]["successful_requests"] == 2
    assert summary["performance"]["input_tokens"] == 12
    assert summary["performance"]["output_tokens"] == 10
    assert len(rows) == 2
    for row in rows:
        assert row["model"] == MODEL_ID
        assert row["raw_output"] == ANSWER
        assert row["score"] == 1.0
        assert row["usage_available"] is True
        assert row["input_tokens"] == 6 and row["output_tokens"] == 5
        assert row["request_id"] == "fixture-request"
        assert row["error"] is None
        assert row["finish_reason"]
        assert (row["ttft_ms"] is not None) is stream
    assert events[0]["event"] == "run_started"
    assert events[0]["api"] == api
    assert events[-1]["event"] == "run_completed"
    assert _read_json(output / "run_state.json")["status"] == "completed"


@pytest.mark.parametrize(
    ("provider", "api", "header", "prefix"),
    [
        ("openai", "responses", "authorization", "Bearer "),
        ("xai", "responses", "authorization", "Bearer "),
        ("anthropic", "messages", "x-api-key", ""),
        ("gemini", "generate-content", "x-goog-api-key", ""),
    ],
    ids=["openai", "xai", "anthropic", "gemini"],
)
def test_provider_auth_and_secret_free_artifacts(provider, api, header, prefix, tmp_path) -> None:
    state = MockState()
    dataset, output = tmp_path / "custom.jsonl", tmp_path / "auth"
    make_dataset(dataset)
    secret = "fixture-private-key-do-not-save"
    with run_local_server(state) as base:
        result = invoke_run(
            base_url=f"{base}/gateway/proxy/v1",
            api=api,
            dataset=dataset,
            output_dir=output,
            extra=["--provider", provider, "--api-key", secret],
        )
    assert result.exit_code == 0, (result.output, result.exception)
    assert len(state.calls) == 2
    for call in state.calls:
        assert call["path"].startswith("/gateway/proxy/v1/")
        assert call["headers"][header] == prefix + secret
        if api == "messages":
            assert call["headers"]["anthropic-version"] == "2023-06-01"
    assert secret not in result.output
    for path in output.rglob("*"):
        if path.is_file():
            assert secret not in path.read_text(encoding="utf-8"), path


def test_responses_reasoning_summary_is_not_scored_as_answer(tmp_path) -> None:
    state = MockState()
    state.reasoning = True
    dataset, output = tmp_path / "custom.jsonl", tmp_path / "reasoning"
    make_dataset(dataset)
    with run_local_server(state) as base:
        result = invoke_run(
            base_url=f"{base}/v1", api="responses", dataset=dataset, output_dir=output
        )
    assert result.exit_code == 0, (result.output, result.exception)
    assert _read_json(output / "summary.json")["quality"]["sample_mean_score"] == 1.0
    assert all(row["raw_output"] == ANSWER for row in _read_rows(output / "raw_results.jsonl"))


def test_resume_uses_saved_model_and_mismatches_send_no_requests(tmp_path) -> None:
    state = MockState()
    dataset, output = tmp_path / "custom.jsonl", tmp_path / "resume"
    make_dataset(dataset)
    with run_local_server(state) as base:
        first = invoke_run(
            base_url=f"{base}/proxy/v1", api="chat", dataset=dataset, output_dir=output
        )
        assert first.exit_code == 0, (first.output, first.exception)
        original_rows = (output / "raw_results.jsonl").read_bytes()
        state.calls.clear()
        # The saved run supplies --model only when continuing an already identified run.
        resumed = CliRunner().invoke(app, ["run", "--resume", str(output)])
        assert resumed.exit_code == 0, (resumed.output, resumed.exception)
        assert state.calls == []
        assert (output / "raw_results.jsonl").read_bytes() == original_rows
        assert _read_json(output / "summary.json")["model"] == MODEL_ID
        for option, value in (("--api", "responses"), ("--model", "another-model")):
            mismatch = CliRunner().invoke(app, ["run", "--resume", str(output), option, value])
            assert mismatch.exit_code != 0, mismatch.output
            assert state.calls == []
            assert (output / "raw_results.jsonl").read_bytes() == original_rows


@pytest.mark.parametrize("command", ["run", "eval", "stress"])
def test_new_evaluation_requires_model_before_any_http(command, tmp_path) -> None:
    state = MockState()
    with run_local_server(state) as base:
        result = CliRunner().invoke(
            app, [command, "--base-url", f"{base}/v1", "--output-dir", str(tmp_path / command)]
        )
    assert result.exit_code != 0
    assert "model" in result.output.lower(), (result.output, result.exception)
    assert state.calls == []


@pytest.mark.parametrize("api", APIS)
def test_http_failure_has_infra_exit_and_error_artifacts(api, tmp_path) -> None:
    state = MockState()
    state.force_status = 500
    dataset, output = tmp_path / "custom.jsonl", tmp_path / "failed"
    make_dataset(dataset)
    with run_local_server(state) as base:
        result = invoke_run(base_url=f"{base}/v1", api=api, dataset=dataset, output_dir=output)
    assert result.exit_code == 4, (result.output, result.exception)
    assert len(state.calls) == 2
    assert all(call["method"] == "POST" for call in state.calls)
    for name in ARTIFACTS:
        assert (output / name).is_file(), name
    summary = _read_json(output / "summary.json")
    assert summary["performance"]["failed_requests"] == 2
    assert summary["quality"]["quality_valid"] is False
    for row in _read_rows(output / "raw_results.jsonl"):
        assert row["model"] == MODEL_ID
        assert row["error"]
        assert row["http_status"] == 500
        assert row["attempts"] == 1


def test_truncation_is_visible_in_summary(tmp_path) -> None:
    state = MockState()
    state.answer, state.finish_reason = '{"answer":"7"', "length"
    dataset, output = tmp_path / "custom.jsonl", tmp_path / "truncated"
    make_dataset(dataset)
    with run_local_server(state) as base:
        result = invoke_run(base_url=f"{base}/v1", api="chat", dataset=dataset, output_dir=output)
    assert result.exit_code == 0, (result.output, result.exception)
    summary = _read_json(output / "summary.json")
    assert summary["performance"]["truncated_responses"] == 2
    assert summary["quality"]["quality_valid"] is False
