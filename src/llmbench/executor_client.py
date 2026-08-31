from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx


class RemoteExecutorClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout, transport=transport)

    async def __aenter__(self) -> RemoteExecutorClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self._client.aclose()

    async def submit(self, payload: dict[str, Any], *, ephemeral_key: str) -> dict[str, Any]:
        request = {**payload, "ephemeral_key": ephemeral_key}
        response = await self._client.post(f"{self.base_url}/v1/jobs", json=request)
        response.raise_for_status()
        return response.json()

    async def get(self, job_id: str) -> dict[str, Any]:
        response = await self._client.get(f"{self.base_url}/v1/jobs/{job_id}")
        response.raise_for_status()
        return response.json()

    async def artifacts(self, job_id: str) -> dict[str, Any]:
        response = await self._client.get(f"{self.base_url}/v1/jobs/{job_id}/artifacts")
        response.raise_for_status()
        return response.json()

    async def cancel(self, job_id: str) -> dict[str, Any]:
        response = await self._client.post(f"{self.base_url}/v1/jobs/{job_id}/cancel")
        response.raise_for_status()
        return response.json()

    async def wait(
        self,
        job_id: str,
        *,
        timeout: float = 600.0,
        poll_interval: float = 0.5,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            job = await self.get(job_id)
            if job["status"] in {"completed", "failed", "cancelled"}:
                return job
            await asyncio.sleep(poll_interval)
        raise TimeoutError(f"executor job did not finish within {timeout} seconds: {job_id}")
