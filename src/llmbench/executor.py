from __future__ import annotations

import asyncio
import contextlib
import json
import os
import tempfile
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

TERMINAL_STATES = {"completed", "failed", "cancelled"}
Emit = Callable[[str, dict[str, Any]], Awaitable[None]]


@dataclass(slots=True)
class ExecutorConfig:
    allowed_images: set[str]
    work_dir: Path
    allow_insecure: bool = False
    allow_network: bool = False
    network_name: str | None = None
    network_proxy: str | None = None
    allowed_domains: tuple[str, ...] = ()
    docker_bin: str = "docker"
    cpus: float = 1.0
    memory: str = "2g"
    pids_limit: int = 128
    timeout_seconds: float = 300.0
    output_limit_bytes: int = 1_000_000


class ExecutorBackend(Protocol):
    async def run(
        self,
        payload: dict[str, Any],
        ephemeral_key: str,
        emit: Emit,
    ) -> dict[str, Any]: ...


class DockerBackend:
    def __init__(self, config: ExecutorConfig) -> None:
        self.config = config

    def docker_command(self, payload: dict[str, Any], env_file: Path) -> list[str]:
        image = str(payload.get("image") or "")
        if image not in self.config.allowed_images:
            raise ValueError(f"executor image is not allowlisted: {image}")
        command = payload.get("command")
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(part, str) for part in command)
        ):
            raise ValueError("executor command must be a non-empty string array")
        network_requested = bool(payload.get("network"))
        if network_requested and not self.config.allow_network:
            raise ValueError("network access is disabled by executor policy")
        if network_requested and (
            not self.config.network_name
            or not self.config.network_proxy
            or not self.config.allowed_domains
        ):
            raise ValueError(
                "network tasks require an isolated network, proxy, and domain allowlist"
            )
        network = self.config.network_name if network_requested else "none"
        command_prefix = [
            self.config.docker_bin,
            "run",
            "--rm",
            "--read-only",
            "--network",
            network,
            "--pids-limit",
            str(self.config.pids_limit),
            "--memory",
            self.config.memory,
            "--cpus",
            str(self.config.cpus),
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=256m",
            "--env-file",
            str(env_file),
        ]
        if network_requested:
            command_prefix.extend(
                [
                    "-e",
                    f"HTTP_PROXY={self.config.network_proxy}",
                    "-e",
                    f"HTTPS_PROXY={self.config.network_proxy}",
                    "-e",
                    "LLMBENCH_ALLOWED_DOMAINS=" + ",".join(self.config.allowed_domains),
                ]
            )
        return [*command_prefix, image, *command]

    async def run(
        self,
        payload: dict[str, Any],
        ephemeral_key: str,
        emit: Emit,
    ) -> dict[str, Any]:
        if "\n" in ephemeral_key or "\r" in ephemeral_key:
            raise ValueError("ephemeral key contains forbidden newline characters")
        self.config.work_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.config.work_dir, 0o700)
        descriptor, raw_path = tempfile.mkstemp(
            prefix="llmbench-secret-", suffix=".env", dir=self.config.work_dir
        )
        env_path = Path(raw_path)
        try:
            os.chmod(env_path, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(f"LLMBENCH_TASK_KEY={ephemeral_key}\n")
            command = self.docker_command(payload, env_path)
            await emit("container_started", {"image": payload["image"]})
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=self.config.timeout_seconds
                )
            except TimeoutError:
                process.kill()
                await process.communicate()
                raise TimeoutError("executor task exceeded its time limit") from None
            limit = self.config.output_limit_bytes
            output = stdout[:limit].decode("utf-8", errors="replace")
            error_output = stderr[:limit].decode("utf-8", errors="replace")
            output = output.replace(ephemeral_key, "[REDACTED]")
            error_output = error_output.replace(ephemeral_key, "[REDACTED]")
            return {
                "exit_code": process.returncode,
                "stdout": output,
                "stderr": error_output,
                "truncated": len(stdout) > limit or len(stderr) > limit,
            }
        finally:
            env_path.unlink(missing_ok=True)


@dataclass(slots=True)
class JobRecord:
    id: str
    status: str
    request: dict[str, Any]
    events: list[dict[str, Any]] = field(default_factory=list)
    artifact: dict[str, Any] | None = None
    error: str | None = None

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "request": self.request,
            "error": self.error,
        }


class JobStore:
    def __init__(self, backend: ExecutorBackend) -> None:
        self.backend = backend
        self.records: dict[str, JobRecord] = {}
        self.tasks: dict[str, asyncio.Task] = {}
        self.queues: dict[str, asyncio.Queue[dict[str, Any]]] = {}

    async def submit(self, payload: dict[str, Any], ephemeral_key: str) -> JobRecord:
        job_id = uuid.uuid4().hex
        sanitized = {key: value for key, value in payload.items() if "key" not in key.casefold()}
        record = JobRecord(id=job_id, status="queued", request=sanitized)
        self.records[job_id] = record
        self.queues[job_id] = asyncio.Queue()
        self.tasks[job_id] = asyncio.create_task(self._execute(record, payload, ephemeral_key))
        return record

    async def _event(self, record: JobRecord, event: str, **fields: Any) -> None:
        payload = {"event": event, "job_id": record.id, **fields}
        record.events.append(payload)
        await self.queues[record.id].put(payload)

    async def _execute(
        self, record: JobRecord, payload: dict[str, Any], ephemeral_key: str
    ) -> None:
        try:
            record.status = "running"
            await self._event(record, "job_started")

            async def emit(event: str, fields: dict[str, Any]) -> None:
                safe = {
                    key: str(value).replace(ephemeral_key, "[REDACTED]")
                    for key, value in fields.items()
                    if "key" not in key.casefold()
                }
                await self._event(record, event, **safe)

            artifact = await self.backend.run(payload, ephemeral_key, emit)
            serialized = json.dumps(artifact, ensure_ascii=False).replace(
                ephemeral_key, "[REDACTED]"
            )
            record.artifact = json.loads(serialized)
            record.status = "completed"
            await self._event(record, "job_completed")
        except asyncio.CancelledError:
            record.status = "cancelled"
            await self._event(record, "job_cancelled")
            raise
        except Exception as exc:
            record.status = "failed"
            record.error = str(exc).replace(ephemeral_key, "[REDACTED]")
            await self._event(record, "job_failed", error=record.error)

    async def cancel(self, job_id: str) -> JobRecord:
        record = self.get(job_id)
        task = self.tasks.get(job_id)
        if task and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            if record.status not in TERMINAL_STATES:
                record.status = "cancelled"
                await self._event(record, "job_cancelled")
        return record

    def get(self, job_id: str) -> JobRecord:
        if job_id not in self.records:
            raise KeyError(job_id)
        return self.records[job_id]

    async def event_stream(self, job_id: str) -> AsyncIterator[str]:
        record = self.get(job_id)
        index = 0
        while True:
            while index < len(record.events):
                event = record.events[index]
                index += 1
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            if record.status in TERMINAL_STATES:
                return
            await asyncio.sleep(0.05)


def create_executor_app(config: ExecutorConfig, backend: ExecutorBackend | None = None):
    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import StreamingResponse
    except ImportError as exc:
        raise RuntimeError("Install quanttrio-llmbench[executor] to serve the executor") from exc

    # FastAPI resolves postponed annotations against module globals.
    globals()["Request"] = Request

    app = FastAPI(title="llmbench executor", version="1")
    store = JobStore(backend or DockerBackend(config))
    app.state.job_store = store

    @app.post("/v1/jobs", status_code=202)
    async def submit_job(request: Request, payload: dict[str, Any]) -> dict[str, Any]:
        forwarded = request.headers.get("x-forwarded-proto")
        if not config.allow_insecure and request.url.scheme != "https" and forwarded != "https":
            raise HTTPException(status_code=400, detail="TLS is required")
        ephemeral_key = payload.pop("ephemeral_key", None)
        if not isinstance(ephemeral_key, str) or not ephemeral_key:
            raise HTTPException(status_code=400, detail="ephemeral_key is required")
        record = await store.submit(payload, ephemeral_key)
        return record.public()

    @app.get("/v1/jobs/{job_id}")
    async def get_job(job_id: str) -> dict[str, Any]:
        try:
            return store.get(job_id).public()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc

    @app.get("/v1/jobs/{job_id}/events")
    async def job_events(job_id: str):
        try:
            store.get(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc
        return StreamingResponse(store.event_stream(job_id), media_type="text/event-stream")

    @app.post("/v1/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str) -> dict[str, Any]:
        try:
            return (await store.cancel(job_id)).public()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc

    @app.get("/v1/jobs/{job_id}/artifacts")
    async def get_artifacts(job_id: str) -> dict[str, Any]:
        try:
            record = store.get(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc
        if record.status != "completed":
            raise HTTPException(status_code=409, detail="job is not completed")
        return record.artifact or {}

    return app
