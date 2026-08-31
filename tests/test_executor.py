from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from llmbench.cli import app as cli_app
from llmbench.executor import DockerBackend, ExecutorConfig, JobStore, create_executor_app
from llmbench.executor_client import RemoteExecutorClient


class FakeBackend:
    async def run(self, payload, ephemeral_key: str, emit):
        await emit("backend_log", {"message": f"using {ephemeral_key}"})
        return {"exit_code": 0, "stdout": f"done {ephemeral_key}", "request": payload}


class SlowBackend:
    async def run(self, payload, ephemeral_key: str, emit):
        await emit("waiting", {})
        await asyncio.sleep(10)
        return {}


class FailingBackend:
    async def run(self, payload, ephemeral_key: str, emit):
        raise RuntimeError(f"failed with {ephemeral_key}")


def config(tmp_path, *, allow_insecure: bool = True) -> ExecutorConfig:
    return ExecutorConfig(
        allowed_images={"sandbox:test"},
        work_dir=tmp_path,
        allow_insecure=allow_insecure,
    )


@pytest.mark.asyncio
async def test_executor_api_redacts_ephemeral_key(tmp_path) -> None:
    app = create_executor_app(config(tmp_path), backend=FakeBackend())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://executor") as client:
        response = await client.post(
            "/v1/jobs",
            json={
                "ephemeral_key": "top-secret",
                "image": "sandbox:test",
                "command": ["-c", "print('ok')"],
            },
        )
        assert response.status_code == 202
        job_id = response.json()["id"]
        for _ in range(20):
            status = (await client.get(f"/v1/jobs/{job_id}")).json()
            if status["status"] == "completed":
                break
            await asyncio.sleep(0.01)
        artifact = (await client.get(f"/v1/jobs/{job_id}/artifacts")).json()
        assert artifact["stdout"] == "done [REDACTED]"
        record = app.state.job_store.get(job_id)
        serialized = json.dumps(
            {"record": record.public(), "events": record.events, "artifact": record.artifact}
        )
        assert "top-secret" not in serialized
        events = await client.get(f"/v1/jobs/{job_id}/events")
        assert "job_completed" in events.text
        assert "top-secret" not in events.text


@pytest.mark.asyncio
async def test_remote_executor_client_lifecycle(tmp_path) -> None:
    app = create_executor_app(config(tmp_path), backend=FakeBackend())
    transport = httpx.ASGITransport(app=app)
    async with RemoteExecutorClient("http://executor", transport=transport) as client:
        submitted = await client.submit(
            {"image": "sandbox:test", "command": ["test"]},
            ephemeral_key="temporary",
        )
        completed = await client.wait(submitted["id"], timeout=1, poll_interval=0.01)
        assert completed["status"] == "completed"
        artifact = await client.artifacts(submitted["id"])
        assert artifact["exit_code"] == 0


def test_executor_smoke_cli(monkeypatch) -> None:
    class Client:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            return None

        async def submit(self, payload, *, ephemeral_key: str):
            assert ephemeral_key == "temporary"
            return {"id": "job-1"}

        async def wait(self, job_id: str, *, timeout: float):
            return {"id": job_id, "status": "completed"}

        async def artifacts(self, job_id: str):
            return {"exit_code": 0, "stdout": "ok", "job_id": job_id}

    monkeypatch.setattr("llmbench.executor_client.RemoteExecutorClient", Client)
    result = CliRunner().invoke(
        cli_app,
        [
            "executor",
            "smoke",
            "--executor-url",
            "http://executor",
            "--ephemeral-key",
            "temporary",
            "--image",
            "sandbox:test",
        ],
    )
    assert result.exit_code == 0, result.exception
    assert '"stdout": "ok"' in result.stdout


@pytest.mark.asyncio
async def test_executor_requires_tls_by_default(tmp_path) -> None:
    app = create_executor_app(config(tmp_path, allow_insecure=False), backend=FakeBackend())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://executor") as client:
        response = await client.post(
            "/v1/jobs",
            json={"ephemeral_key": "secret", "image": "sandbox:test", "command": ["test"]},
        )
    assert response.status_code == 400

    insecure_app = create_executor_app(config(tmp_path), backend=FakeBackend())
    insecure_transport = httpx.ASGITransport(app=insecure_app)
    async with httpx.AsyncClient(
        transport=insecure_transport, base_url="http://executor"
    ) as client:
        missing_key = await client.post(
            "/v1/jobs", json={"image": "sandbox:test", "command": ["test"]}
        )
    assert missing_key.status_code == 400


@pytest.mark.asyncio
async def test_executor_cancel_conflict_and_not_found_routes(tmp_path) -> None:
    app = create_executor_app(config(tmp_path), backend=SlowBackend())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://executor") as client:
        assert (await client.get("/v1/jobs/missing")).status_code == 404
        assert (await client.get("/v1/jobs/missing/events")).status_code == 404
        assert (await client.post("/v1/jobs/missing/cancel")).status_code == 404
        assert (await client.get("/v1/jobs/missing/artifacts")).status_code == 404
        created = await client.post(
            "/v1/jobs",
            json={"ephemeral_key": "secret", "image": "sandbox:test", "command": ["test"]},
        )
        job_id = created.json()["id"]
        assert (await client.get(f"/v1/jobs/{job_id}/artifacts")).status_code == 409
        cancelled = await client.post(f"/v1/jobs/{job_id}/cancel")
        assert cancelled.json()["status"] == "cancelled"


@pytest.mark.asyncio
async def test_job_failure_redacts_secret(tmp_path) -> None:
    store = JobStore(FailingBackend())
    record = await store.submit({"image": "sandbox:test"}, "secret-value")
    await store.tasks[record.id]
    assert record.status == "failed"
    assert record.error == "failed with [REDACTED]"
    assert "secret-value" not in json.dumps(record.events)


def test_docker_command_enforces_isolation_and_allowlist(tmp_path) -> None:
    backend = DockerBackend(config(tmp_path))
    command = backend.docker_command(
        {"image": "sandbox:test", "command": ["-c", "print('ok')"]},
        Path("/tmp/secret.env"),
    )
    assert "--read-only" in command
    assert command[command.index("--network") + 1] == "none"
    assert command[command.index("--cap-drop") + 1] == "ALL"
    assert "no-new-privileges" in command
    with pytest.raises(ValueError, match="allowlisted"):
        backend.docker_command({"image": "unknown", "command": ["test"]}, Path("/tmp/secret.env"))
    with pytest.raises(ValueError, match="network"):
        backend.docker_command(
            {"image": "sandbox:test", "command": ["test"], "network": True},
            Path("/tmp/secret.env"),
        )
    with pytest.raises(ValueError, match="command"):
        backend.docker_command({"image": "sandbox:test", "command": []}, Path("/tmp/secret.env"))
    network_config = config(tmp_path)
    network_config.allow_network = True
    network_config.network_name = "isolated"
    network_config.network_proxy = "http://proxy:3128"
    network_config.allowed_domains = (".example.com",)
    network_command = DockerBackend(network_config).docker_command(
        {"image": "sandbox:test", "command": ["test"], "network": True},
        Path("/tmp/secret.env"),
    )
    assert network_command[network_command.index("--network") + 1] == "isolated"
    assert "HTTP_PROXY=http://proxy:3128" in network_command
    assert "LLMBENCH_ALLOWED_DOMAINS=.example.com" in network_command


@pytest.mark.asyncio
async def test_docker_backend_run_redacts_output_and_removes_secret_file(
    tmp_path, monkeypatch
) -> None:
    class Process:
        returncode = 0

        async def communicate(self):
            return b"secret-value output", b"secret-value error"

    captured = {}

    async def create_process(*command, **kwargs):
        captured["command"] = command
        return Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    backend = DockerBackend(config(tmp_path))
    events = []

    async def emit(event, fields):
        events.append((event, fields))

    artifact = await backend.run(
        {"image": "sandbox:test", "command": ["-c", "print('ok')"]},
        "secret-value",
        emit,
    )
    assert artifact["stdout"] == "[REDACTED] output"
    assert artifact["stderr"] == "[REDACTED] error"
    assert events[0][0] == "container_started"
    env_path = Path(captured["command"][captured["command"].index("--env-file") + 1])
    assert not env_path.exists()
    with pytest.raises(ValueError, match="newline"):
        await backend.run(
            {"image": "sandbox:test", "command": ["test"]},
            "bad\nsecret",
            emit,
        )


@pytest.mark.asyncio
async def test_docker_backend_timeout_kills_process(tmp_path, monkeypatch) -> None:
    class Process:
        returncode = None

        def __init__(self) -> None:
            self.calls = 0
            self.killed = False

        async def communicate(self):
            self.calls += 1
            if self.calls == 1:
                await asyncio.sleep(1)
            return b"", b""

        def kill(self) -> None:
            self.killed = True

    process = Process()

    async def create_process(*args, **kwargs):
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    settings = config(tmp_path)
    settings.timeout_seconds = 0.001
    backend = DockerBackend(settings)

    async def emit(event, fields):
        return None

    with pytest.raises(TimeoutError, match="time limit"):
        await backend.run(
            {"image": "sandbox:test", "command": ["test"]},
            "secret",
            emit,
        )
    assert process.killed is True
