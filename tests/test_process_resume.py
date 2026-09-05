"""Real CLI processes must resume persisted work after an interrupted SSE request."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

MODEL_ID = "fixture-model"
DATASET = "process-resume"
QUESTION_COUNT = 4
SOURCE = Path(__file__).resolve().parents[1] / "src"


class ServerState:
    def __init__(self, pause_after: int) -> None:
        self.pause_call = pause_after + 1
        self.paused = threading.Event()
        self.release = threading.Event()
        self.calls: list[dict] = []


def _event(payload: dict) -> bytes:
    return f"event: {payload['type']}\ndata: {json.dumps(payload)}\n\n".encode()


@contextmanager
def _server(state: ServerState) -> Iterator[str]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: object) -> None:
            pass

        def do_POST(self) -> None:
            self.connection.settimeout(10)
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            match = re.search(r"\[question:(q-\d+)\]", json.dumps(body.get("input")))
            question_id = match[1] if match else "unknown"
            call_number = len(state.calls) + 1
            request_id = f"request-{question_id}-call-{call_number}"
            state.calls.append(
                {
                    "path": self.path,
                    "body": body,
                    "question_id": question_id,
                    "request_id": request_id,
                }
            )
            answer = f"answer-{question_id}"
            response = {
                "id": f"resp-{question_id}-call-{call_number}",
                "object": "response",
                "status": "completed",
                "model": MODEL_ID,
                "output": [
                    {
                        "type": "message",
                        "id": f"msg-{question_id}",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": answer, "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            }
            try:
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.send_header("x-request-id", request_id)
                self.end_headers()
                self.wfile.write(
                    _event(
                        {
                            "type": "response.created",
                            "response": {**response, "status": "in_progress", "output": []},
                        }
                    )
                )
                self.wfile.write(
                    _event(
                        {
                            "type": "response.output_text.delta",
                            "item_id": f"msg-{question_id}",
                            "output_index": 0,
                            "content_index": 0,
                            "delta": answer[:3] if call_number == state.pause_call else answer,
                        }
                    )
                )
                self.wfile.flush()
                if call_number == state.pause_call:
                    # Stop mid-stream only once; the resumed request can finish immediately.
                    state.paused.set()
                    state.release.wait(timeout=20)
                    return
                self.wfile.write(_event({"type": "response.completed", "response": response}))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass  # The interrupted child has closed its connection.

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        state.release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive()


@contextmanager
def _child(command: list[str], log: Path) -> Iterator[subprocess.Popen]:
    # A fresh environment excludes hosted credentials and proxies from this offline test.
    environment = {"PYTHONPATH": str(SOURCE), "PYTHONIOENCODING": "utf-8"}
    with log.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(
            [sys.executable, "-m", "llmbench.cli", *command],
            cwd=log.parent,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
        )
        try:
            yield process
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)


def _finish(command: list[str], log: Path) -> None:
    with _child(command, log) as process:
        assert process.wait(timeout=20) == 0, log.read_text(encoding="utf-8")


def _rows(raw: bytes) -> list[dict]:
    return [json.loads(line) for line in raw.splitlines()]


@pytest.mark.skipif(os.name != "posix", reason="SIGINT/SIGKILL child semantics require POSIX")
@pytest.mark.parametrize(
    ("signal_name", "completed_count"),
    [("SIGINT", 1), ("SIGKILL", 2)],
    ids=["sigint-after-one", "sigkill-after-two"],
)
def test_cli_process_interrupt_and_resume(
    tmp_path: Path, signal_name: str, completed_count: int
) -> None:
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text(
        "".join(
            json.dumps(
                {
                    "id": f"q-{index}",
                    "dataset": DATASET,
                    "type": "exact_match",
                    "question": f"[question:q-{index}] Return exactly answer-q-{index}.",
                    "answer": f"answer-q-{index}",
                }
            )
            + "\n"
            for index in range(QUESTION_COUNT)
        ),
        encoding="utf-8",
    )
    output = tmp_path / "result"
    raw_path = output / "raw_results.jsonl"
    state = ServerState(completed_count)
    with _server(state) as base_url:
        command = [
            "run",
            "--base-url",
            base_url,
            "--api",
            "responses",
            "--model",
            MODEL_ID,
            "--dataset",
            str(dataset),
            "--limit",
            str(QUESTION_COUNT),
            "--mode",
            "quality",
            "--concurrency",
            "1",
            "--max-tokens",
            "64",
            "--stream",
            "--checkpoint-every",
            "1",
            "--retries",
            "0",
            "--timeout",
            "15",
            "--output-dir",
            str(output),
        ]
        first_log = tmp_path / "interrupted.log"
        with _child(command, first_log) as process:
            assert state.paused.wait(timeout=15), first_log.read_text(encoding="utf-8")
            assert process.poll() is None
            persisted = raw_path.read_bytes()
            completed = _rows(persisted)
            assert len(completed) == completed_count
            completed_ids = {row["question_id"] for row in completed}
            assert completed_ids == {f"q-{index}" for index in range(completed_count)}
            process.send_signal(getattr(signal, signal_name))
            returncode = process.wait(timeout=10)
            assert returncode != 0, first_log.read_text(encoding="utf-8")
            if signal_name == "SIGKILL":
                assert returncode == -signal.SIGKILL
        assert raw_path.read_bytes() == persisted
        assert len(state.calls) == completed_count + 1
        previous_calls = len(state.calls)
        state.release.set()

        resume = [*command, "--resume", str(output)]
        _finish(resume, tmp_path / "resumed.log")
        final_raw = raw_path.read_bytes()
        assert final_raw.startswith(persisted)  # Includes original answers and request IDs.
        rows = _rows(final_raw)
        expected_keys = {(DATASET, f"q-{index}", 1) for index in range(QUESTION_COUNT)}
        assert len(rows) == QUESTION_COUNT
        assert {(row["dataset"], row["question_id"], row["sample_id"]) for row in rows} == (
            expected_keys
        )
        assert completed_ids.isdisjoint(
            call["question_id"] for call in state.calls[previous_calls:]
        )
        counts = Counter(call["question_id"] for call in state.calls)
        assert counts == Counter(
            {f"q-{index}": 2 if index == completed_count else 1 for index in range(QUESTION_COUNT)}
        )
        for call in state.calls:
            assert call["path"] == "/v1/responses"
            assert call["body"]["model"] == MODEL_ID
            assert call["body"]["stream"] is True
        for row in rows:
            assert row["raw_output"] == row["gold_answer"]
            assert row["raw_output"] == f"answer-{row['question_id']}"
            assert row["request_id"].startswith(f"request-{row['question_id']}-call-")
            assert row["score"] == 1.0
            assert row["error"] is None
            assert row["ttft_ms"] is not None
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        assert summary["evaluation"]["complete"] is True
        assert summary["performance"]["successful_requests"] == QUESTION_COUNT
        calls_after_completion = list(state.calls)

        _finish(resume, tmp_path / "already-completed.log")
        assert state.calls == calls_after_completion
        assert raw_path.read_bytes() == final_raw
        run_state = json.loads((output / "run_state.json").read_text(encoding="utf-8"))
        assert run_state["status"] == "completed"
        assert run_state["completed"] == run_state["total"] == QUESTION_COUNT
