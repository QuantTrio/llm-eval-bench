from __future__ import annotations

import json

import httpx
import pytest

from llmbench.client import OpenAICompatibleClient


@pytest.mark.asyncio
async def test_model_discovery_and_json_completion() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "model-a"}, {"id": "model-b"}]})
        assert request.headers["authorization"] == "Bearer secret"
        body = json.loads(request.content)
        assert body["stream"] is False
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"answer":"A"}'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4},
            },
        )

    async with OpenAICompatibleClient(
        base_url="http://test/v1/", api_key="secret", transport=httpx.MockTransport(handler)
    ) as client:
        selected, models = await client.resolve_model(None)
        assert selected == "model-a"
        assert models == ["model-a", "model-b"]
        result = await client.complete(
            model=selected,
            messages=[{"role": "user", "content": "test"}],
            temperature=0,
            top_p=1,
            max_tokens=16,
            stream=False,
        )
    assert result.content == '{"answer":"A"}'
    assert result.input_tokens == 10
    assert result.output_tokens == 4


@pytest.mark.asyncio
async def test_streaming_metrics_and_usage() -> None:
    stream = "\n\n".join(
        [
            'data: {"choices":[{"delta":{"content":"{\\"answer\\""},"finish_reason":null}]}',
            'data: {"choices":[{"delta":{"content":":\\"B\\"}"},"finish_reason":"stop"}]}',
            'data: {"choices":[],"usage":{"prompt_tokens":8,"completion_tokens":3}}',
            "data: [DONE]",
        ]
    )

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=stream, headers={"content-type": "text/event-stream"})

    async with OpenAICompatibleClient(
        base_url="http://test/v1", api_key="EMPTY", transport=httpx.MockTransport(handler)
    ) as client:
        result = await client.complete(
            model="model-a",
            messages=[{"role": "user", "content": "test"}],
            temperature=0,
            top_p=1,
            max_tokens=16,
            stream=True,
        )
    assert result.content == '{"answer":"B"}'
    assert result.ttft_ms is not None
    assert result.tpot_ms is not None
    assert result.output_tokens == 3


@pytest.mark.asyncio
async def test_retries_429_then_succeeds() -> None:
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, request=request, text="rate limited")
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "A"}, "finish_reason": "stop"}]},
        )

    async with OpenAICompatibleClient(
        base_url="http://test/v1",
        api_key="EMPTY",
        retries=1,
        retry_backoff=0,
        transport=httpx.MockTransport(handler),
    ) as client:
        result = await client.complete(
            model="model-a",
            messages=[{"role": "user", "content": "test"}],
            temperature=0,
            top_p=1,
            max_tokens=16,
            stream=False,
        )
    assert attempts == 2
    assert result.error is None
    assert result.attempts == 2
