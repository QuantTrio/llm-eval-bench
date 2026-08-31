from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from .schemas import CompletionResult


class OpenAICompatibleClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout: float = 120.0,
        retries: int = 2,
        retry_backoff: float = 2.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.retries = retries
        self.retry_backoff = retry_backoff
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            headers=headers,
            timeout=httpx.Timeout(timeout),
            transport=transport,
        )

    async def __aenter__(self) -> OpenAICompatibleClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def list_models(self) -> list[str]:
        response = await self._client.get(f"{self.base_url}/models")
        response.raise_for_status()
        payload = response.json()
        models = [str(item["id"]) for item in payload.get("data", []) if item.get("id")]
        if not models:
            raise RuntimeError(f"No model IDs returned by {self.base_url}/models")
        return models

    async def resolve_model(self, requested: str | None) -> tuple[str, list[str]]:
        models = await self.list_models()
        if requested:
            if requested not in models:
                raise ValueError(
                    f"Model '{requested}' is not exposed by the server. "
                    f"Available: {', '.join(models)}"
                )
            return requested, models
        return models[0], models

    async def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        stream: bool,
        seed: int | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> CompletionResult:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if seed is not None:
            body["seed"] = seed
        if extra_body:
            protected = {"model", "messages", "stream"} & set(extra_body)
            if protected:
                names = ", ".join(sorted(protected))
                raise ValueError(f"request extra body cannot override protected fields: {names}")
            body.update(extra_body)
        if stream:
            body["stream_options"] = {"include_usage": True}

        overall_started = time.perf_counter()
        for attempt in range(1, self.retries + 2):
            started = time.perf_counter()
            try:
                if stream:
                    result = await self._stream_completion(body, started)
                else:
                    result = await self._json_completion(body, started)
                result.attempts = attempt
                result.attempt_latency_ms = result.latency_ms
                result.latency_ms = (time.perf_counter() - overall_started) * 1000
                return result
            except httpx.TimeoutException as exc:
                result = self._error_result(started, "timeout", exc, attempts=attempt)
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                kind = "http_429" if status == 429 else f"http_{status}"
                result = self._error_result(started, kind, exc, status=status, attempts=attempt)
                if status < 500 and status != 429:
                    return result
            except (httpx.HTTPError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                result = self._error_result(started, "protocol_error", exc, attempts=attempt)

            if attempt <= self.retries:
                await asyncio.sleep(self.retry_backoff * (2 ** (attempt - 1)))
        result.attempt_latency_ms = result.latency_ms
        result.latency_ms = (time.perf_counter() - overall_started) * 1000
        return result

    async def _json_completion(self, body: dict[str, Any], started: float) -> CompletionResult:
        response = await self._client.post(f"{self.base_url}/chat/completions", json=body)
        response.raise_for_status()
        payload = response.json()
        choice = payload["choices"][0]
        usage = payload.get("usage") or {}
        latency_ms = (time.perf_counter() - started) * 1000
        return CompletionResult(
            content=str((choice.get("message") or {}).get("content") or ""),
            latency_ms=latency_ms,
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            finish_reason=choice.get("finish_reason"),
        )

    async def _stream_completion(self, body: dict[str, Any], started: float) -> CompletionResult:
        content: list[str] = []
        usage: dict[str, Any] = {}
        finish_reason: str | None = None
        ttft_ms: float | None = None
        async with self._client.stream(
            "POST", f"{self.base_url}/chat/completions", json=body
        ) as response:
            response.raise_for_status()
            async for payload in self._sse_payloads(response):
                if payload.get("usage"):
                    usage = payload["usage"]
                choices = payload.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = (choice.get("delta") or {}).get("content") or ""
                if delta:
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - started) * 1000
                    content.append(str(delta))
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
        latency_ms = (time.perf_counter() - started) * 1000
        output_tokens = int(usage.get("completion_tokens") or 0)
        tpot_ms = None
        if ttft_ms is not None and output_tokens > 1:
            tpot_ms = max(0.0, latency_ms - ttft_ms) / (output_tokens - 1)
        return CompletionResult(
            content="".join(content),
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=output_tokens,
            finish_reason=finish_reason,
        )

    @staticmethod
    async def _sse_payloads(response: httpx.Response) -> AsyncIterator[dict[str, Any]]:
        async for line in response.aiter_lines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            if raw == "[DONE]":
                break
            if raw:
                yield json.loads(raw)

    @staticmethod
    def _error_result(
        started: float,
        kind: str,
        exc: Exception,
        *,
        status: int | None = None,
        attempts: int,
    ) -> CompletionResult:
        return CompletionResult(
            latency_ms=(time.perf_counter() - started) * 1000,
            error=str(exc),
            error_type=kind,
            http_status=status,
            attempts=attempts,
        )
