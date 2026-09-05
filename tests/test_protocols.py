import json
from copy import deepcopy

import pytest

from llmbench import protocols


def completion_body(api, messages):
    return protocols.build_completion_body(
        api,
        model="vision-model",
        messages=messages,
        temperature=0,
        top_p=1,
        max_tokens=64,
        stream=False,
        seed=42,
    )


@pytest.mark.parametrize("api", protocols.SUPPORTED_APIS)
@pytest.mark.parametrize("image_count", [1, 2])
def test_native_image_mapping_preserves_text_order_and_image_detail(api, image_count) -> None:
    content = [{"type": "text", "text": "Question\nA. White\nB. Black\nReturn JSON."}]
    urls = []
    for index in range(image_count):
        url = f"data:image/{'png' if index == 0 else 'jpeg'};base64,cG5n"
        urls.append(url)
        content.append({"type": "image_url", "image_url": {"url": url, "detail": "high"}})
        content.append({"type": "text", "text": f"After image {index + 1}."})
    messages = [
        {"role": "system", "content": "Answer accurately."},
        {"role": "user", "content": deepcopy(content)},
    ]
    body = completion_body(api, messages)
    assert messages[1]["content"] == content
    if api == "chat":
        assert body["messages"] == messages
        assert body["seed"] == 42
        return
    if api == "responses":
        assert body["instructions"] == "Answer accurately."
        blocks = body["input"][0]["content"]
        expected = [{"type": "input_text", "text": content[0]["text"]}]
        for index, url in enumerate(urls):
            expected.extend(
                [
                    {"type": "input_image", "image_url": url, "detail": "high"},
                    {"type": "input_text", "text": f"After image {index + 1}."},
                ]
            )
    elif api == "messages":
        assert body["system"] == "Answer accurately."
        blocks = body["messages"][0]["content"]
        expected = [content[0]]
        for index, url in enumerate(urls):
            expected.extend(
                [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": url.split(";")[0].removeprefix("data:"),
                            "data": "cG5n",
                        },
                    },
                    {"type": "text", "text": f"After image {index + 1}."},
                ]
            )
    else:
        assert body["systemInstruction"] == {"parts": [{"text": "Answer accurately."}]}
        assert body["contents"][0]["role"] == "user"
        blocks = body["contents"][0]["parts"]
        expected = [{"text": content[0]["text"]}]
        for index, url in enumerate(urls):
            expected.extend(
                [
                    {
                        "inlineData": {
                            "mimeType": url.split(";")[0].removeprefix("data:"),
                            "data": "cG5n",
                        }
                    },
                    {"text": f"After image {index + 1}."},
                ]
            )
    assert blocks == expected


@pytest.mark.parametrize("api", protocols.SUPPORTED_APIS)
@pytest.mark.parametrize(
    "block",
    [
        None,
        "base64 image",
        {"type": "audio", "data": "cG5n"},
        {"type": "input_image", "image_url": "data:image/png;base64,cG5n"},
        {"type": "text", "text": {"image_url": "cG5n"}},
        {"type": "image_url"},
        {"type": "image_url", "image_url": "data:image/png;base64,cG5n"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,!!!"}},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,"}},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,cG5n", "detail": "bogus"},
        },
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,cG5n", "detail": {}},
        },
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,cG5n", "extra": True},
        },
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,cG5n"},
            "text": "ambiguous",
        },
    ],
)
def test_malformed_or_unknown_image_blocks_are_rejected(api, block) -> None:
    with pytest.raises(ValueError):
        completion_body(api, [{"role": "user", "content": [block]}])


@pytest.mark.parametrize("api", protocols.SUPPORTED_APIS)
def test_text_request_fields_remain_unchanged(api) -> None:
    messages = [
        {"role": "system", "content": "Instruction one."},
        {"role": "developer", "content": "Instruction two."},
        {"role": "user", "content": "Question?"},
        {"role": "assistant", "content": "Prior answer."},
    ]
    body = completion_body(api, messages)
    if api == "chat":
        assert body == {
            "model": "vision-model",
            "messages": messages,
            "max_tokens": 64,
            "stream": False,
            "temperature": 0,
            "top_p": 1,
            "seed": 42,
        }
    elif api == "responses":
        assert body == {
            "model": "vision-model",
            "input": messages[2:],
            "max_output_tokens": 64,
            "stream": False,
            "store": False,
            "temperature": 0,
            "top_p": 1,
            "instructions": "Instruction one.\nInstruction two.",
        }
    elif api == "messages":
        assert body == {
            "model": "vision-model",
            "messages": messages[2:],
            "max_tokens": 64,
            "stream": False,
            "temperature": 0,
            "top_p": 1,
            "system": "Instruction one.\nInstruction two.",
        }
    else:
        assert body == {
            "contents": [
                {"role": "user", "parts": [{"text": "Question?"}]},
                {"role": "model", "parts": [{"text": "Prior answer."}]},
            ],
            "generationConfig": {"maxOutputTokens": 64, "temperature": 0, "topP": 1, "seed": 42},
            "systemInstruction": {"parts": [{"text": "Instruction one.\nInstruction two."}]},
        }


@pytest.mark.parametrize("api", protocols.SUPPORTED_APIS)
def test_native_image_count_and_aggregate_size_bounds(api, monkeypatch) -> None:
    block = {"type": "image_url", "image_url": {"url": "data:image/png;base64,cG5n"}}
    monkeypatch.setattr(protocols, "MAX_IMAGES", 1)
    with pytest.raises(ValueError, match="image count or total byte"):
        completion_body(api, [{"role": "user", "content": [block, block]}])
    monkeypatch.setattr(protocols, "MAX_TOTAL_IMAGE_BYTES", 2)
    with pytest.raises(ValueError, match="image count or total byte"):
        completion_body(api, [{"role": "user", "content": [block]}])


@pytest.mark.parametrize(
    "api,usage,expected",
    [
        (
            "chat",
            {
                "prompt_tokens": 8,
                "completion_tokens": 6,
                "completion_tokens_details": {"reasoning_tokens": 4},
                "prompt_tokens_details": {"cached_tokens": 2},
            },
            (8, 6, 4, 2),
        ),
        (
            "responses",
            {
                "input_tokens": 8,
                "output_tokens": 6,
                "output_tokens_details": {"reasoning_tokens": 4},
                "input_tokens_details": {"cached_tokens": 2},
            },
            (8, 6, 4, 2),
        ),
        (
            "messages",
            {
                "input_tokens": 3,
                "output_tokens": 6,
                "cache_read_input_tokens": 2,
                "cache_creation_input_tokens": 3,
            },
            (8, 6, None, 2),
        ),
        (
            "generate-content",
            {
                "promptTokenCount": 8,
                "candidatesTokenCount": 2,
                "thoughtsTokenCount": 4,
                "cachedContentTokenCount": 2,
            },
            (8, 6, 4, 2),
        ),
    ],
)
def test_token_counts_include_reasoning_and_cached_input(
    api: str, usage: dict, expected: tuple
) -> None:
    result = protocols.usage_details(api, usage)
    assert (
        tuple(
            result[key]
            for key in ("input_tokens", "output_tokens", "reasoning_tokens", "cached_input_tokens")
        )
        == expected
    )
    assert result["usage_available"]


@pytest.mark.parametrize("api", protocols.SUPPORTED_APIS)
def test_unknown_usage_and_reasoning_are_not_measured_zero(api: str) -> None:
    result = protocols.usage_details(api, {})
    assert result["usage_available"] is False
    assert result["reasoning_tokens"] is None
    assert result["cached_input_tokens"] is None


@pytest.mark.parametrize(
    "api,payload",
    [
        ("chat", {"choices": [{"message": {"content": "partial"}, "finish_reason": "length"}]}),
        (
            "responses",
            {
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "output": [
                    {"type": "message", "content": [{"type": "output_text", "text": "partial"}]}
                ],
            },
        ),
        (
            "messages",
            {"content": [{"type": "text", "text": "partial"}], "stop_reason": "max_tokens"},
        ),
        (
            "generate-content",
            {
                "candidates": [
                    {"content": {"parts": [{"text": "partial"}]}, "finishReason": "MAX_TOKENS"}
                ]
            },
        ),
    ],
)
def test_all_protocols_normalize_truncation(api: str, payload: dict) -> None:
    parsed = protocols.parse_json_completion(api, payload)
    assert parsed["finish_reason"] == "length"
    assert parsed["content"] == "partial"


@pytest.mark.parametrize(
    "api,payload",
    [
        ("chat", {}),
        ("chat", {"choices": []}),
        ("chat", {"choices": [{"message": {"content": "A"}}]}),
        ("responses", {"status": "failed", "error": {"message": "bad"}}),
        ("responses", {"status": "in_progress", "output": []}),
        ("responses", {"status": "incomplete", "output": []}),
        (
            "responses",
            {
                "status": "incomplete",
                "incomplete_details": {"reason": "content_filter"},
                "output": [],
            },
        ),
        ("messages", {"content": []}),
        ("generate-content", {"promptFeedback": {"blockReason": "SAFETY"}}),
    ],
)
def test_incomplete_or_malformed_json_cannot_succeed(api: str, payload: dict) -> None:
    with pytest.raises(ValueError):
        protocols.parse_json_completion(api, payload)


@pytest.mark.parametrize("key", sorted(protocols.PROTECTED_EXTRA_BODY_KEYS))
def test_extra_body_cannot_override_identity_state_tools_or_sampling(key: str) -> None:
    with pytest.raises(ValueError, match="protected"):
        protocols.validate_extra_body("chat", {key: None})


@pytest.mark.parametrize(
    "key", ["maxOutputTokens", "temperature", "topP", "seed", "candidateCount"]
)
def test_gemini_extra_cannot_override_core_generation_config(key: str) -> None:
    with pytest.raises(ValueError, match="protected"):
        protocols.validate_extra_body("generate-content", {"generationConfig": {key: 1}})


def test_extra_body_accepts_provider_reasoning_parameters() -> None:
    protocols.validate_extra_body("responses", {"reasoning": {"effort": "high"}})
    protocols.validate_extra_body("messages", {"thinking": {"type": "adaptive"}})
    protocols.validate_extra_body(
        "generate-content", {"generationConfig": {"thinkingConfig": {"thinkingBudget": 1024}}}
    )


def test_sse_multiline_events_comments_and_event_reset() -> None:
    lines = [
        ": heartbeat",
        "event: first",
        "data: {",
        'data: "value": 1}',
        "",
        'data: {"value": 2}',
        "",
        "data: [DONE]",
        "",
    ]
    assert list(protocols.iter_sse_events(lines)) == [
        ("first", {"value": 1}),
        (None, {"value": 2}),
        ("[DONE]", {}),
    ]


@pytest.mark.parametrize(
    "lines",
    [["data: {not-json}", ""], ["data: []", ""], ['data: {"type": "done"}'], [": heartbeat", ""]],
)
def test_malformed_or_unterminated_sse_raises(lines: list[str]) -> None:
    with pytest.raises(ValueError):
        list(protocols.iter_sse_events(lines))


def test_responses_terminal_does_not_duplicate_delta_answer() -> None:
    state = protocols.StreamState()
    protocols.apply_stream_payload(
        "responses", state, None, {"type": "response.output_text.delta", "delta": "A"}
    )
    protocols.apply_stream_payload(
        "responses", state, None, {"type": "response.output_text.done", "text": "A"}
    )
    protocols.apply_stream_payload(
        "responses",
        state,
        None,
        {
            "type": "response.completed",
            "response": {
                "id": "response-1",
                "status": "completed",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "A"}]}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        },
    )
    assert state.content == "A"
    assert state.complete
    assert state.usage_available
    assert state.response_id == "response-1"


def test_messages_usage_merges_start_and_delta() -> None:
    state = protocols.StreamState()
    frames = [
        {
            "type": "message_start",
            "message": {"id": "msg-1", "usage": {"input_tokens": 7, "output_tokens": 1}},
        },
        {"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": "hidden"}},
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "A"}},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "max_tokens"},
            "usage": {"output_tokens": 6},
        },
        {"type": "message_stop"},
    ]
    for frame in frames:
        protocols.apply_stream_payload("messages", state, None, frame)
    assert state.content == "A"
    assert state.finish_reason == "length"
    assert state.complete
    assert state.usage == {"input_tokens": 7, "output_tokens": 6}
    assert state.saw_reasoning
    assert state.response_id == "msg-1"
    assert "hidden" not in json.dumps(state.__dict__)


def test_gemini_model_prefix_removed_and_name_url_encoded() -> None:
    assert (
        protocols.completion_path("generate-content", "models/model-a", stream=False)
        == "/models/model-a:generateContent"
    )
    assert (
        protocols.completion_path("generate-content", "model?key=value", stream=True)
        == "/models/model%3Fkey%3Dvalue:streamGenerateContent?alt=sse"
    )
