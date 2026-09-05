import json

import httpx
import pytest

from llmbench.client import OpenAICompatibleClient, _parse_retry_after

CHAT = {
    "choices": [{"message": {"content": "A"}, "finish_reason": "stop"}],
    "usage": {
        "prompt_tokens": 8,
        "completion_tokens": 3,
        "completion_tokens_details": {"reasoning_tokens": 0},
    },
}
RESPONSES = {
    "id": "resp_1",
    "status": "completed",
    "output": [
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "secret"}]},
        {"type": "message", "content": [{"type": "output_text", "text": "A"}]},
    ],
    "usage": {
        "input_tokens": 8,
        "output_tokens": 3,
        "output_tokens_details": {"reasoning_tokens": 0},
        "input_tokens_details": {"cached_tokens": 2},
    },
}
MESSAGES = {
    "id": "msg_1",
    "content": [{"type": "thinking", "thinking": "secret"}, {"type": "text", "text": "A"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 8, "output_tokens": 3},
}
GEMINI = {
    "candidates": [
        {
            "content": {"parts": [{"thought": True, "text": "secret"}, {"text": "A"}]},
            "finishReason": "STOP",
        }
    ],
    "usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": 3, "thoughtsTokenCount": 2},
}


def _sse(*frames: dict | str) -> str:
    return "".join(
        f"data: {json.dumps(frame) if isinstance(frame, dict) else frame}\n\n" for frame in frames
    )


def _stream(api: str) -> str:
    if api == "chat":
        return _sse(
            {"choices": [{"delta": {"reasoning_content": "secret"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": "A"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            {"choices": [], "usage": CHAT["usage"]},
            "[DONE]",
        )
    if api == "responses":
        return _sse(
            {"type": "response.reasoning_summary_text.delta", "delta": "secret"},
            {"type": "response.output_text.delta", "delta": "A"},
            {"type": "response.output_text.done", "text": "A"},
            {"type": "response.completed", "response": RESPONSES},
        )
    if api == "messages":
        return _sse(
            {
                "type": "message_start",
                "message": {"usage": {"input_tokens": 8, "output_tokens": 1}},
            },
            {
                "type": "content_block_delta",
                "delta": {"type": "thinking_delta", "thinking": "secret"},
            },
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "A"}},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 3},
            },
            {"type": "message_stop"},
        )
    return _sse(GEMINI)


@pytest.mark.asyncio
async def test_explicit_model_required_without_discovery() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        pytest.fail("model selection must never call /models")

    async with OpenAICompatibleClient(
        base_url="http://test/v1/",
        api_key="",
        transport=httpx.MockTransport(handler),
    ) as client:
        for model in (None, "", "   "):
            with pytest.raises(ValueError, match="--model"):
                await client.resolve_model(model)
        assert await client.resolve_model("exact/model-id") == ("exact/model-id", [])


@pytest.mark.asyncio
async def test_list_models_is_separate_diagnostic() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json={"data": [{"id": "m1"}, {"id": "m2"}]})

    async with OpenAICompatibleClient(
        base_url="http://test/v1/",
        api_key="",
        transport=httpx.MockTransport(handler),
    ) as client:
        assert await client.list_models() == ["m1", "m2"]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "api,provider,header,path,payload",
    [
        ("chat", None, "authorization", "/proxy/v1/chat/completions", CHAT),
        ("responses", "xai", "authorization", "/proxy/v1/responses", RESPONSES),
        ("messages", None, "x-api-key", "/proxy/v1/messages", MESSAGES),
        ("generate-content", None, "x-goog-api-key", "/proxy/v1/models/model-id", GEMINI),
    ],
)
async def test_native_protocol_requests_and_results(
    api: str,
    provider: str | None,
    header: str,
    path: str,
    payload: dict,
    stream: bool,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path.startswith(path)
        assert request.headers[header] == (
            "Bearer secret-key" if header == "authorization" else "secret-key"
        )
        if api == "messages":
            assert request.headers["anthropic-version"] == "2023-06-01"
        body = json.loads(request.content)
        if api == "generate-content":
            assert body["generationConfig"] == {"maxOutputTokens": 16, "seed": 42}
            assert body["systemInstruction"] == {"parts": [{"text": "system text"}]}
            assert body["contents"] == [{"role": "user", "parts": [{"text": "test"}]}]
            suffix = ":streamGenerateContent" if stream else ":generateContent"
            assert request.url.path.endswith(suffix)
            assert request.url.params.get("alt") == ("sse" if stream else None)
        else:
            assert body["model"] == "model-id"
            assert "temperature" not in body and "top_p" not in body
            if api != "chat":
                assert "seed" not in body
            if api == "responses":
                assert body["store"] is False
                assert body["instructions"] == "system text"
                assert body["max_output_tokens"] == 16
            elif api == "messages":
                assert body["system"] == "system text"
        if stream:
            return httpx.Response(200, text=_stream(api), headers={"x-request-id": "req-1"})
        return httpx.Response(200, json=payload, headers={"x-request-id": "req-1"})

    async with OpenAICompatibleClient(
        base_url="http://test/proxy/v1/",
        api_key="secret-key",
        api=api,
        provider=provider,
        transport=httpx.MockTransport(handler),
    ) as client:
        result = await client.complete(
            model="model-id",
            messages=[
                {"role": "system", "content": "system text"},
                {"role": "user", "content": "test"},
            ],
            max_tokens=16,
            seed=42,
            stream=stream,
        )
    assert result.error is None
    assert result.content == "A"
    assert result.request_id == "req-1"
    assert result.input_tokens == 8
    assert result.output_tokens == (5 if api == "generate-content" else 3)
    assert result.usage_available
    assert result.raw_response is not None
    assert (result.ttft_ms is not None) == stream
    # These fixtures emit reasoning even where usage incorrectly claims zero reasoning tokens.
    # The adapter must not publish a text-only TPOT for such contradictory accounting.
    assert result.tpot_ms is None


@pytest.mark.asyncio
@pytest.mark.parametrize("api", ["chat", "responses", "messages", "generate-content"])
async def test_premature_eof_preserves_partial_answer_as_error(api: str) -> None:
    first = {
        "chat": {"choices": [{"delta": {"content": "partial"}}]},
        "responses": {"type": "response.output_text.delta", "delta": "partial"},
        "messages": {
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "partial"},
        },
        "generate-content": {"candidates": [{"content": {"parts": [{"text": "partial"}]}}]},
    }[api]
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, text=_sse(first), headers={"x-request-id": "broken-stream"})

    async with OpenAICompatibleClient(
        base_url="http://test/v1",
        api_key="",
        api=api,
        retries=2,
        transport=httpx.MockTransport(handler),
    ) as client:
        result = await client.complete(model="m", messages=[], stream=True)
    assert result.error_type == "protocol_error"
    assert result.content == "partial"
    assert result.ttft_ms is not None
    assert result.tpot_ms is None
    assert result.request_id == "broken-stream"
    assert calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_error_response_redacts_credential_recursively(stream: bool) -> None:
    secret = "my-secret-key"
    error = {"type": "error", "error": {"message": f"echo {secret}", secret: [secret]}}

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=_sse(error) if stream else json.dumps(error))

    async with OpenAICompatibleClient(
        base_url="http://test/v1",
        api_key=secret,
        api="messages",
        transport=httpx.MockTransport(handler),
    ) as client:
        result = await client.complete(model="m", messages=[], stream=stream)
    assert result.error_type == "protocol_error"
    assert secret not in result.error
    assert secret not in json.dumps(result.raw_response)
    assert "REDACTED" in json.dumps(result.raw_response)


@pytest.mark.asyncio
@pytest.mark.parametrize("status,retried", [(401, False), (400, False), (429, True), (503, True)])
async def test_retry_policy_and_retry_after(status: int, retried: bool, monkeypatch) -> None:
    attempts = 0
    sleeps = []

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr("llmbench.client.asyncio.sleep", sleep)

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(
                status,
                json={"error": {"message": "failure"}},
                headers={"retry-after": "6", "x-request-id": "failure-id"},
            )
        return httpx.Response(200, json=CHAT)

    async with OpenAICompatibleClient(
        base_url="http://test/v1",
        api_key="",
        retries=1,
        retry_backoff=0,
        transport=httpx.MockTransport(handler),
    ) as client:
        result = await client.complete(model="m", messages=[])
        assert client._retry_delay(1, 1e9) == 30.0
    assert attempts == (2 if retried else 1)
    assert sleeps == ([6.0] if retried else [])
    assert result.attempts == attempts
    assert result.latency_ms >= result.attempt_latency_ms
    if retried:
        assert result.error is None
    else:
        assert result.http_status == status
        assert result.request_id == "failure-id"


@pytest.mark.asyncio
async def test_gemini_extra_thinking_config_merges_without_losing_budget() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        config = json.loads(request.content)["generationConfig"]
        assert config == {
            "maxOutputTokens": 17,
            "seed": 42,
            "thinkingConfig": {"thinkingBudget": 0},
        }
        return httpx.Response(200, json=GEMINI)

    async with OpenAICompatibleClient(
        base_url="http://test/v1beta",
        api_key="",
        api="generate-content",
        transport=httpx.MockTransport(handler),
    ) as client:
        result = await client.complete(
            model="m",
            messages=[],
            max_tokens=17,
            seed=42,
            extra_body={"generationConfig": {"thinkingConfig": {"thinkingBudget": 0}}},
        )
    assert result.error is None


@pytest.mark.parametrize("value", [None, "garbage", "NaN", "Inf"])
def test_invalid_retry_after_is_ignored(value: str | None) -> None:
    assert _parse_retry_after(value) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_tail", [False, True])
async def test_gemini_reads_trailing_usage_or_error_after_finish(bad_tail: bool) -> None:
    first = {
        "responseId": "gemini-id",
        "candidates": [{"content": {"parts": [{"text": "A"}]}, "finishReason": "STOP"}],
    }
    tail = (
        {"error": {"message": "midstream failure"}}
        if bad_tail
        else {
            "usageMetadata": {
                "promptTokenCount": 8,
                "candidatesTokenCount": 3,
                "thoughtsTokenCount": 0,
            }
        }
    )

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=_sse(first, tail))

    async with OpenAICompatibleClient(
        base_url="http://test/v1beta",
        api_key="",
        api="generate-content",
        transport=httpx.MockTransport(handler),
    ) as client:
        result = await client.complete(model="m", messages=[], stream=True)
    assert result.content == "A"
    assert result.request_id == "gemini-id"
    if bad_tail:
        assert result.error_type == "protocol_error"
        assert result.tpot_ms is None
    else:
        assert result.error is None
        assert result.usage_available
        assert result.input_tokens == 8
        assert result.output_tokens == 3
        assert result.tpot_ms is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "api,payload,response_id",
    [
        ("responses", RESPONSES, "resp_1"),
        ("messages", MESSAGES, "msg_1"),
        ("generate-content", {**GEMINI, "responseId": "gemini-id"}, "gemini-id"),
    ],
)
async def test_body_response_id_fallback(api: str, payload: dict, response_id: str) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    async with OpenAICompatibleClient(
        base_url="http://test/v1",
        api_key="",
        api=api,
        transport=httpx.MockTransport(handler),
    ) as client:
        result = await client.complete(model="m", messages=[])
    assert result.request_id == response_id


@pytest.mark.asyncio
@pytest.mark.parametrize("api", ["chat", "responses", "messages", "generate-content"])
async def test_reasoning_only_stream_has_no_answer_ttft(api: str) -> None:
    frames = {
        "chat": [
            {"choices": [{"delta": {"reasoning_content": "hidden"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "length"}]},
            "[DONE]",
        ],
        "responses": [
            {"type": "response.reasoning_summary_text.delta", "delta": "hidden"},
            {
                "type": "response.incomplete",
                "response": {
                    "status": "incomplete",
                    "output": [],
                    "incomplete_details": {"reason": "max_output_tokens"},
                },
            },
        ],
        "messages": [
            {
                "type": "content_block_delta",
                "delta": {"type": "thinking_delta", "thinking": "hidden"},
            },
            {"type": "message_delta", "delta": {"stop_reason": "max_tokens"}},
            {"type": "message_stop"},
        ],
        "generate-content": [
            {
                "candidates": [
                    {
                        "content": {"parts": [{"thought": True, "text": "hidden"}]},
                        "finishReason": "MAX_TOKENS",
                    }
                ]
            }
        ],
    }[api]

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=_sse(*frames))

    async with OpenAICompatibleClient(
        base_url="http://test/v1",
        api_key="",
        api=api,
        transport=httpx.MockTransport(handler),
    ) as client:
        result = await client.complete(model="m", messages=[], stream=True)
    assert result.error is None
    assert result.content == ""
    assert result.finish_reason == "length"
    assert result.ttft_ms is None
    assert result.tpot_ms is None


@pytest.mark.asyncio
@pytest.mark.parametrize("reasoning", [None, 0, 2])
async def test_tpot_requires_explicit_zero_reasoning_count(reasoning: int | None) -> None:
    usage = {"prompt_tokens": 8, "completion_tokens": 3}
    if reasoning is not None:
        usage["completion_tokens_details"] = {"reasoning_tokens": reasoning}
    stream = _sse(
        {"choices": [{"delta": {"content": "A"}, "finish_reason": "stop"}]},
        {"choices": [], "usage": usage},
        "[DONE]",
    )

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=stream)

    async with OpenAICompatibleClient(
        base_url="http://test/v1",
        api_key="",
        transport=httpx.MockTransport(handler),
    ) as client:
        result = await client.complete(model="m", messages=[], stream=True)
    assert result.error is None
    assert (result.tpot_ms is not None) == (reasoning == 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [httpx.ConnectError, httpx.ReadTimeout])
async def test_transient_transport_errors_are_retried(failure: type[Exception]) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise failure("transient", request=request)
        return httpx.Response(200, json=CHAT)

    async with OpenAICompatibleClient(
        base_url="http://test/v1",
        api_key="",
        retries=1,
        retry_backoff=0,
        transport=httpx.MockTransport(handler),
    ) as client:
        result = await client.complete(model="m", messages=[])
    assert result.error is None
    assert result.attempts == 2


@pytest.mark.asyncio
async def test_redirect_is_reported_without_switching_endpoint() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(307, headers={"location": "http://another-host/v1/messages"})

    async with OpenAICompatibleClient(
        base_url="http://test/v1",
        api_key="secret",
        transport=httpx.MockTransport(handler),
    ) as client:
        result = await client.complete(model="m", messages=[], stream=True)
    assert result.error_type == "http_307"
    assert calls == 1
