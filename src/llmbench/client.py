from __future__ import annotations

import asyncio
import math
import time
from collections.abc import AsyncIterator
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from . import protocols
from .schemas import CompletionResult, EmbeddingResult

PROTECTED_EXTRA_BODY_KEYS = protocols.PROTECTED_EXTRA_BODY_KEYS
_PROTOCOL_ERRORS = (ValueError, KeyError, TypeError, IndexError, AttributeError)


def _build_headers(api: str, api_key: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api == "messages":
        headers["anthropic-version"] = "2023-06-01"
        if api_key:
            headers["x-api-key"] = api_key
    elif api_key:
        if api == "generate-content":
            headers["x-goog-api-key"] = api_key
        else:
            headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _extract_request_id(
    response: httpx.Response,
    payload: dict[str, Any] | None = None,
) -> str | None:
    for key in ("x-request-id", "x-amzn-requestid", "request-id", "x-goog-request-id"):
        if response.headers.get(key):
            return response.headers[key]
    if isinstance(payload, dict):
        response_id = payload.get("id") or payload.get("responseId")
        return response_id if isinstance(response_id, str) else None
    return None


def _redact_value(value: Any, secret: str) -> Any:
    if not secret:
        return value
    if isinstance(value, str):
        return value.replace(secret, "***REDACTED***")
    if isinstance(value, list):
        return [_redact_value(item, secret) for item in value]
    if isinstance(value, dict):
        return {
            _redact_value(key, secret): _redact_value(item, secret) for key, item in value.items()
        }
    return value


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        delay = float(value)
    except ValueError:
        try:
            delay = parsedate_to_datetime(value).timestamp() - time.time()
        except (ValueError, TypeError, OverflowError):
            return None
    return max(0.0, delay) if math.isfinite(delay) else None


class OpenAICompatibleClient:
    """One pooled HTTP transport for four stateless inference protocols."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        api: str = "chat",
        provider: str | None = None,
        timeout: float = 120.0,
        retries: int = 2,
        retry_backoff: float = 2.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api = protocols.normalize_api(api)
        self.provider = provider
        self.retries = retries
        self.retry_backoff = retry_backoff
        self._api_key = api_key
        self._client = httpx.AsyncClient(
            headers=_build_headers(self.api, api_key),
            timeout=httpx.Timeout(timeout),
            transport=transport,
            # Benchmark concurrency is bounded by the runner, not httpx's default 100 limit.
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=100),
        )

    async def __aenter__(self) -> OpenAICompatibleClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def list_models(self) -> list[str]:
        """Explicit diagnostic only; running a benchmark never discovers or chooses a model."""
        try:
            response = await self._client.get(f"{self.base_url}/models")
            response.raise_for_status()
            payload = response.json()
            models: list[str] = []
            for item in payload.get("data", []):
                if isinstance(item, dict) and item.get("id"):
                    models.append(str(item["id"]))
            for item in payload.get("models", []):
                name = item.get("name") or item.get("id") if isinstance(item, dict) else item
                if isinstance(name, str) and name:
                    models.append(name.removeprefix("models/"))
            if not models:
                raise ValueError("no model IDs returned by /models")
            return list(dict.fromkeys(models))
        except (httpx.HTTPError, *_PROTOCOL_ERRORS) as exc:
            raise RuntimeError(_redact_value(str(exc), self._api_key)) from None

    async def resolve_model(self, requested: str | None) -> tuple[str, list[str]]:
        if not requested or not requested.strip():
            raise ValueError("an explicit model ID is required; pass --model MODEL_ID")
        return requested, []

    async def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int = 1024,
        stream: bool = False,
        seed: int | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> CompletionResult:
        await self.resolve_model(model)
        body = protocols.build_completion_body(
            self.api,
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=stream,
            seed=seed,
        )
        if extra_body:
            protocols.validate_extra_body(self.api, extra_body)
            for key, value in extra_body.items():
                if key == "generationConfig":
                    body[key].update(value)
                else:
                    body[key] = value
        url = self.base_url + protocols.completion_path(self.api, model=model, stream=stream)
        overall_started = time.perf_counter()
        result = CompletionResult(error="no completion attempt", error_type="protocol_error")
        for attempt in range(1, self.retries + 2):
            started = time.perf_counter()
            if stream:
                result, retry_after = await self._stream_completion(url, body, started)
            else:
                result, retry_after = await self._json_completion(url, body, started)
            result.attempts = attempt
            result.attempt_latency_ms = result.latency_ms
            if (
                result.error is None
                or attempt > self.retries
                or not self._is_retryable_error(result)
            ):
                break
            await asyncio.sleep(self._retry_delay(attempt, retry_after))
        result.latency_ms = (time.perf_counter() - overall_started) * 1000
        return result

    async def embed(self, *, model: str, inputs: list[str]) -> EmbeddingResult:
        started = time.perf_counter()
        try:
            response = await self._client.post(
                f"{self.base_url}/embeddings",
                json={"model": model, "input": inputs},
            )
            response.raise_for_status()
            payload = response.json()
            ordered = sorted(payload.get("data") or [], key=lambda item: item.get("index", 0))
            vectors = [[float(value) for value in item["embedding"]] for item in ordered]
            if len(vectors) != len(inputs):
                raise ValueError(
                    f"embedding endpoint returned {len(vectors)} vectors for {len(inputs)} inputs"
                )
            return EmbeddingResult(
                vectors=vectors,
                latency_ms=(time.perf_counter() - started) * 1000,
                input_tokens=int((payload.get("usage") or {}).get("prompt_tokens") or 0),
            )
        except (httpx.HTTPError, *_PROTOCOL_ERRORS) as exc:
            return EmbeddingResult(
                latency_ms=(time.perf_counter() - started) * 1000,
                error=_redact_value(str(exc), self._api_key),
                error_type="timeout"
                if isinstance(exc, httpx.TimeoutException)
                else "embedding_error",
            )

    async def _json_completion(
        self,
        url: str,
        body: dict[str, Any],
        started: float,
    ) -> tuple[CompletionResult, float | None]:
        response = None
        payload = None
        try:
            response = await self._client.post(url, json=body)
            response.raise_for_status()
            payload = response.json()
            parsed = protocols.parse_json_completion(self.api, payload)
            return CompletionResult(
                content=_redact_value(parsed["content"], self._api_key),
                latency_ms=(time.perf_counter() - started) * 1000,
                finish_reason=parsed["finish_reason"],
                request_id=_redact_value(_extract_request_id(response, payload), self._api_key),
                raw_response=_redact_value(payload, self._api_key),
                **protocols.usage_details(self.api, parsed["usage"]),
            ), None
        except (httpx.HTTPError, *_PROTOCOL_ERRORS) as exc:
            return self._error_result(started, exc, response=response, raw=payload)

    async def _stream_completion(
        self,
        url: str,
        body: dict[str, Any],
        started: float,
    ) -> tuple[CompletionResult, float | None]:
        state = protocols.StreamState()
        ttft_ms = None
        response = None
        try:
            async with self._client.stream("POST", url, json=body) as response:
                if response.is_error:
                    await response.aread()
                response.raise_for_status()
                async for event, payload in self._sse_payloads(response):
                    previous_length = state.content_length
                    protocols.apply_stream_payload(self.api, state, event, payload)
                    if ttft_ms is None and state.content_length > previous_length:
                        ttft_ms = (time.perf_counter() - started) * 1000
                    if state.complete and self.api != "generate-content":
                        break
            if not state.complete:
                raise ValueError("stream ended without terminal marker")
            usage = protocols.usage_details(self.api, state.usage)
            latency_ms = (time.perf_counter() - started) * 1000
            tpot_ms = None
            if (
                usage["usage_available"]
                and usage["reasoning_tokens"] == 0
                and not state.saw_reasoning
                and ttft_ms is not None
                and usage["output_tokens"] > 1
            ):
                tpot_ms = max(0.0, latency_ms - ttft_ms) / (usage["output_tokens"] - 1)
            return CompletionResult(
                content=_redact_value(state.content, self._api_key),
                latency_ms=latency_ms,
                ttft_ms=ttft_ms,
                tpot_ms=tpot_ms,
                finish_reason=state.finish_reason,
                request_id=_redact_value(
                    _extract_request_id(response) or state.response_id,
                    self._api_key,
                ),
                raw_response=_redact_value(state.raw, self._api_key),
                **usage,
            ), None
        except (httpx.HTTPError, *_PROTOCOL_ERRORS) as exc:
            result, retry_after = self._error_result(started, exc, response=response, raw=state.raw)
            result.content = _redact_value(state.content, self._api_key)
            result.ttft_ms = ttft_ms
            result.finish_reason = state.finish_reason
            if result.request_id is None:
                result.request_id = _redact_value(state.response_id, self._api_key)
            return result, retry_after

    async def _sse_payloads(
        self,
        response: httpx.Response,
    ) -> AsyncIterator[tuple[str | None, dict[str, Any]]]:
        event_name: str | None = None
        data_lines: list[str] = []
        seen_data = False
        async for line in response.aiter_lines():
            if not line:
                if data_lines:
                    marker, parsed = protocols.decode_sse_data(data_lines)
                    seen_data = True
                    yield marker or event_name, parsed
                event_name, data_lines = None, []
            elif line.startswith("event:"):
                event_name = line[6:].strip() or None
            elif line.startswith("data:"):
                data_lines.append(line[5:].removeprefix(" "))
        if data_lines:
            raise ValueError("SSE stream ended before frame completion")
        if not seen_data:
            raise ValueError("SSE stream is empty")

    @staticmethod
    def _is_retryable_http(status: int) -> bool:
        return status in {408, 429, 500, 502, 503, 504}

    @classmethod
    def _is_retryable_error(cls, result: CompletionResult) -> bool:
        return result.error_type in {"timeout", "network_error"} or (
            result.http_status is not None and cls._is_retryable_http(result.http_status)
        )

    def _retry_delay(self, attempt: int, retry_after: float | None) -> float:
        delay = self.retry_backoff * (2 ** (attempt - 1)) if retry_after is None else retry_after
        return min(30.0, max(0.0, delay))

    def _error_result(
        self,
        started: float,
        exc: Exception,
        *,
        response: httpx.Response | None = None,
        raw: Any = None,
    ) -> tuple[CompletionResult, float | None]:
        status = response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
        if isinstance(exc, httpx.TimeoutException):
            kind = "timeout"
        elif isinstance(exc, httpx.HTTPStatusError):
            kind = f"http_{status}"
        elif isinstance(exc, httpx.TransportError):
            kind = "network_error"
        else:
            kind = "protocol_error"
        if raw is None and response is not None:
            try:
                raw = response.json()
            except (ValueError, httpx.ResponseNotRead):
                raw = None
        request_id = _extract_request_id(response, raw) if response is not None else None
        retry_after = (
            _parse_retry_after(response.headers.get("retry-after"))
            if response is not None
            else None
        )
        return CompletionResult(
            latency_ms=(time.perf_counter() - started) * 1000,
            error=_redact_value(str(exc), self._api_key),
            error_type=kind,
            http_status=status,
            request_id=_redact_value(request_id, self._api_key),
            raw_response=_redact_value(raw, self._api_key) if isinstance(raw, dict) else None,
        ), retry_after


def discover_models(
    base_url: str,
    api_key: str,
    *,
    api: str = "chat",
    provider: str | None = None,
) -> list[str]:
    async def run() -> list[str]:
        async with OpenAICompatibleClient(
            base_url=base_url,
            api_key=api_key,
            api=api,
            provider=provider,
        ) as client:
            return await client.list_models()

    return asyncio.run(run())
