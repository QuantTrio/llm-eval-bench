"""Stateless request and response adapters; transport lives in client.py."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

from .images import MAX_IMAGES, MAX_TOTAL_IMAGE_BYTES, image_url_block

SUPPORTED_APIS = ("chat", "responses", "messages", "generate-content")
PROTECTED_EXTRA_BODY_KEYS = {
    "model",
    "messages",
    "input",
    "contents",
    "system",
    "systemInstruction",
    "instructions",
    "stream",
    "stream_options",
    "max_tokens",
    "max_completion_tokens",
    "max_output_tokens",
    "temperature",
    "top_p",
    "seed",
    "n",
    "store",
    "previous_response_id",
    "conversation",
    "stateful",
    "background",
    "tools",
    "tool_choice",
    "toolConfig",
    "parallel_tool_calls",
    "cachedContent",
    "context_management",
    "generation_config",
    "system_instruction",
    "cached_content",
    "tool_config",
}
_GEMINI_CORE_CONFIG = {
    "maxOutputTokens",
    "temperature",
    "topP",
    "seed",
    "candidateCount",
    "max_output_tokens",
    "top_p",
    "candidate_count",
}


def normalize_api(api: str) -> str:
    value = (api or "chat").strip().lower().replace("_", "-")
    if value == "generatecontent":
        value = "generate-content"
    if value not in SUPPORTED_APIS:
        raise ValueError(f"unsupported api '{api}', expected one of: {', '.join(SUPPORTED_APIS)}")
    return value


def validate_extra_body(api: str, extra_body: dict[str, Any]) -> None:
    """Keep request identity, sampling and stateless scoring under runner control."""
    if not isinstance(extra_body, dict):
        raise ValueError("request extra body must be a JSON object")
    conflicts = PROTECTED_EXTRA_BODY_KEYS.intersection(extra_body)
    config = extra_body.get("generationConfig")
    if "generationConfig" in extra_body:
        if api != "generate-content" or not isinstance(config, dict):
            raise ValueError("generationConfig must be an object for generate-content")
        conflicts.update(f"generationConfig.{key}" for key in _GEMINI_CORE_CONFIG & config.keys())
    if conflicts:
        raise ValueError(
            "request extra body cannot override protected fields: " + ", ".join(sorted(conflicts))
        )


def _split_messages(messages: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    instructions: list[str] = []
    prompt: list[dict[str, Any]] = []
    for item in messages:
        if not isinstance(item, dict):
            raise ValueError("each message must be an object")
        if item.get("role") in {"system", "developer"}:
            if not isinstance(item.get("content"), str):
                raise ValueError("system and developer instructions must be text")
            instructions.append(item["content"])
        else:
            prompt.append(item)
    return "\n".join(instructions) or None, prompt


def _map_content(content: Any, api: str) -> Any:
    if isinstance(content, str):
        return [{"text": content}] if api == "generate-content" else content
    if not isinstance(content, list) or not content:
        raise ValueError("message content must be text or nonempty canonical content blocks")
    mapped = []
    for block in content:
        if not isinstance(block, dict):
            raise ValueError("content block must be an object")
        if block.get("type") == "text":
            if not isinstance(block.get("text"), str) or set(block) != {"type", "text"}:
                raise ValueError("malformed text content block")
            if api == "responses":
                mapped.append({"type": "input_text", "text": block["text"]})
            elif api == "generate-content":
                mapped.append({"text": block["text"]})
            else:
                mapped.append(dict(block))
        elif block.get("type") == "image_url":
            url, detail = image_url_block(block)
            if api == "chat":
                mapped.append({"type": "image_url", "image_url": dict(block["image_url"])})
            elif api == "responses":
                image = {"type": "input_image", "image_url": url}
                if detail is not None:
                    image["detail"] = detail
                mapped.append(image)
            else:
                header, _, encoded = url.partition(",")
                mime = header.removeprefix("data:").removesuffix(";base64")
                if api == "messages":
                    mapped.append(
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": mime, "data": encoded},
                        }
                    )
                else:
                    mapped.append({"inlineData": {"mimeType": mime, "data": encoded}})
        else:
            raise ValueError("unsupported canonical content block type")
    return mapped


def _map_messages(messages: list[dict[str, Any]], api: str) -> list[dict[str, Any]]:
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    mapped = []
    image_count = total_bytes = 0
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("each message must be an object")
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "image_url":
                    continue
                image_count += 1
                image = block.get("image_url")
                if isinstance(image, dict) and isinstance(image.get("url"), str):
                    encoded = image["url"].partition(",")[2]
                    padding = len(encoded) - len(encoded.rstrip("="))
                    total_bytes += max(0, len(encoded) // 4 * 3 - padding)
                if image_count > MAX_IMAGES or total_bytes > MAX_TOTAL_IMAGE_BYTES:
                    raise ValueError(
                        "images exceed the per-request image count or total byte limit"
                    )
        mapped.append({**message, "content": _map_content(content, api)})
    return mapped


def build_completion_body(
    api: str,
    *,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float | None,
    top_p: float | None,
    max_tokens: int,
    stream: bool,
    seed: int | None,
) -> dict[str, Any]:
    api = normalize_api(api)
    if api == "chat":
        body: dict[str, Any] = {
            "model": model,
            "messages": _map_messages(messages, api),
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if stream:
            body["stream_options"] = {"include_usage": True}
    else:
        instructions, prompt = _split_messages(messages)
        prompt = _map_messages(prompt, api)
        if api == "responses":
            body = {
                "model": model,
                "input": prompt,
                "max_output_tokens": max_tokens,
                "stream": stream,
                "store": False,
            }
            if instructions:
                body["instructions"] = instructions
        elif api == "messages":
            body = {"model": model, "messages": prompt, "max_tokens": max_tokens, "stream": stream}
            if instructions:
                body["system"] = instructions
        else:
            contents = []
            for item in prompt:
                if item.get("role") not in {"user", "assistant"}:
                    raise ValueError("generate-content supports user and assistant messages")
                contents.append(
                    {
                        "role": "model" if item["role"] == "assistant" else "user",
                        "parts": item["content"],
                    }
                )
            body = {"contents": contents, "generationConfig": {"maxOutputTokens": max_tokens}}
            if instructions:
                body["systemInstruction"] = {"parts": [{"text": instructions}]}
    config = body["generationConfig"] if api == "generate-content" else body
    if temperature is not None:
        config["temperature"] = temperature
    if top_p is not None:
        config["topP" if api == "generate-content" else "top_p"] = top_p
    # The runner seed always controls dataset sampling. Only these wire protocols accept it.
    if seed is not None and api in {"chat", "generate-content"}:
        config["seed"] = seed
    return body


def completion_path(api: str, model: str, *, stream: bool) -> str:
    api = normalize_api(api)
    paths = {"chat": "/chat/completions", "responses": "/responses", "messages": "/messages"}
    if api in paths:
        return paths[api]
    model = model.removeprefix("models/")
    method = "streamGenerateContent?alt=sse" if stream else "generateContent"
    return f"/models/{quote(model, safe='')}:{method}"


def _count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def usage_details(api: str, usage: dict[str, Any]) -> dict[str, Any]:
    """Normalize token counters without presenting absent usage as measured zero."""
    reasoning = cached = None
    if api == "chat":
        input_tokens, output_tokens = (
            _count(usage.get("prompt_tokens")),
            _count(usage.get("completion_tokens")),
        )
        reasoning = _count((usage.get("completion_tokens_details") or {}).get("reasoning_tokens"))
        cached = _count((usage.get("prompt_tokens_details") or {}).get("cached_tokens"))
    elif api == "responses":
        input_tokens, output_tokens = (
            _count(usage.get("input_tokens")),
            _count(usage.get("output_tokens")),
        )
        reasoning = _count((usage.get("output_tokens_details") or {}).get("reasoning_tokens"))
        cached = _count((usage.get("input_tokens_details") or {}).get("cached_tokens"))
    elif api == "messages":
        input_tokens, output_tokens = (
            _count(usage.get("input_tokens")),
            _count(usage.get("output_tokens")),
        )
        cached = _count(usage.get("cache_read_input_tokens"))
        if input_tokens is not None:
            input_tokens += (cached or 0) + (_count(usage.get("cache_creation_input_tokens")) or 0)
    else:
        input_tokens, output_tokens = (
            _count(usage.get("promptTokenCount")),
            _count(usage.get("candidatesTokenCount")),
        )
        reasoning = _count(usage.get("thoughtsTokenCount"))
        cached = _count(usage.get("cachedContentTokenCount"))
        # Gemini reports answer and thought tokens separately, unlike Chat/Responses.
        if output_tokens is not None:
            output_tokens += reasoning or 0
    return {
        "input_tokens": input_tokens or 0,
        "output_tokens": output_tokens or 0,
        "reasoning_tokens": reasoning,
        "cached_input_tokens": cached,
        "usage_available": input_tokens is not None and output_tokens is not None,
    }


def _text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError("answer text must be a string")
    return value


def _blocks_text(blocks: Any, *, types: set[str] | None = None) -> tuple[str, str]:
    if not isinstance(blocks, list):
        raise ValueError("content blocks must be a list")
    answer, reasoning = [], []
    for block in blocks:
        if not isinstance(block, dict):
            raise ValueError("content block must be an object")
        kind = block.get("type", "")
        if block.get("thought") is True or kind in {"thinking", "reasoning", "reasoning_text"}:
            reasoning.append(_text(block.get("text") or block.get("thinking")))
        elif types is None or kind in types:
            answer.append(_text(block.get("text")))
    return "".join(answer), "".join(reasoning)


def _finish(value: Any) -> str | None:
    if value in {"length", "max_tokens", "MAX_TOKENS", "max_output_tokens"}:
        return "length"
    return _text(value) or None


def _check_error(payload: dict[str, Any]) -> None:
    if payload.get("error") is not None or payload.get("type") == "error":
        raise ValueError(
            f"API error: {json.dumps(payload.get('error', payload), ensure_ascii=False)}"
        )


def parse_json_completion(api: str, payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("completion response must be a JSON object")
    _check_error(payload)
    reasoning = ""
    if api == "chat":
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ValueError("chat response has no choices")
        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, dict):
            raise ValueError("chat choice has no message")
        content = message.get("content")
        if isinstance(content, list):
            content, reasoning = _blocks_text(content, types={"text", "output_text"})
        else:
            content = _text(content)
        finish = _finish(choice.get("finish_reason"))
    elif api == "responses":
        status = payload.get("status")
        if status not in {"completed", "incomplete"}:
            raise ValueError(f"response did not complete: {status}")
        output = payload.get("output")
        if not isinstance(output, list):
            raise ValueError("Responses output must be a list")
        content = ""
        for item in output:
            if not isinstance(item, dict):
                raise ValueError("Responses output item must be an object")
            if item.get("type") == "message":
                text, _ = _blocks_text(item.get("content"), types={"output_text"})
                content += text
        finish = (
            _finish((payload.get("incomplete_details") or {}).get("reason"))
            if status == "incomplete"
            else "completed"
        )
        if status == "incomplete" and not finish:
            raise ValueError("incomplete response has no reason")
        if status == "incomplete" and finish != "length":
            raise ValueError(f"response incomplete: {finish}")
    elif api == "messages":
        content, reasoning = _blocks_text(payload.get("content"), types={"text"})
        finish = _finish(payload.get("stop_reason"))
    elif api == "generate-content":
        candidates = payload.get("candidates")
        if (
            not isinstance(candidates, list)
            or not candidates
            or not isinstance(candidates[0], dict)
        ):
            raise ValueError("generate-content response has no candidates")
        candidate = candidates[0]
        content, reasoning = _blocks_text((candidate.get("content") or {}).get("parts", []))
        finish = _finish(candidate.get("finishReason"))
    else:
        raise ValueError(f"unsupported api '{api}'")
    if not finish:
        raise ValueError("completion response has no finish reason")
    usage = payload.get("usageMetadata" if api == "generate-content" else "usage") or {}
    if not isinstance(usage, dict):
        raise ValueError("usage must be an object")
    return {
        "content": content,
        "reasoning": reasoning,
        "finish_reason": finish,
        "usage": usage,
        **usage_details(api, usage),
    }


@dataclass
class StreamState:
    chunks: list[str] = field(default_factory=list)
    content_length: int = 0
    saw_reasoning: bool = False
    finish_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    usage_available: bool = False
    raw: dict[str, Any] | None = None
    response_id: str | None = None
    complete: bool = False

    @property
    def content(self) -> str:
        return "".join(self.chunks)

    def append_text(self, text: str) -> None:
        if text:
            self.chunks.append(text)
            self.content_length += len(text)

    def replace_text(self, text: str) -> None:
        self.chunks = [text]
        self.content_length = len(text)


def apply_stream_payload(
    api: str,
    state: StreamState,
    event_name: str | None,
    payload: dict[str, Any],
) -> None:
    if event_name != "[DONE]":
        state.raw = payload
        response_id = payload.get("id") or payload.get("responseId")
        if isinstance(response_id, str):
            state.response_id = response_id
    _check_error(payload)
    event = payload.get("type") or event_name
    if event_name == "[DONE]":
        if api != "chat" or not state.finish_reason:
            raise ValueError("stream terminal marker arrived without a completed answer")
        state.complete = True
        return
    if api == "chat":
        if isinstance(payload.get("usage"), dict):
            state.usage.update(payload["usage"])
        choices = payload.get("choices")
        if not isinstance(choices, list):
            raise ValueError("chat stream payload has no choices")
        if choices:
            choice = choices[0]
            if not isinstance(choice, dict) or not isinstance(choice.get("delta"), dict):
                raise ValueError("chat stream choice has no delta")
            state.append_text(_text(choice["delta"].get("content")))
            state.saw_reasoning |= bool(choice["delta"].get("reasoning_content"))
            if choice.get("finish_reason") is not None:
                state.finish_reason = _finish(choice["finish_reason"])
    elif api == "responses":
        if event in {"response.failed", "response.error"}:
            raise ValueError(f"Responses stream failed: {json.dumps(payload, ensure_ascii=False)}")
        if event == "response.output_text.delta":
            state.append_text(_text(payload.get("delta")))
        elif event and event.startswith("response.reasoning"):
            state.saw_reasoning = True
        elif event in {"response.completed", "response.incomplete"}:
            response = payload.get("response")
            parsed = parse_json_completion(api, response)
            # The terminal response contains the full answer, not another delta.
            if parsed["content"]:
                state.replace_text(parsed["content"])
            state.finish_reason = parsed["finish_reason"]
            state.usage = parsed["usage"]
            state.raw = response
            state.response_id = response.get("id") or state.response_id
            state.complete = True
    elif api == "messages":
        if event == "message_start":
            message = payload.get("message")
            if not isinstance(message, dict):
                raise ValueError("message_start has no message")
            state.usage.update(message.get("usage") or {})
            state.response_id = message.get("id") or state.response_id
        elif event == "content_block_start":
            block = payload.get("content_block") or {}
            if block.get("type") == "text":
                state.append_text(_text(block.get("text")))
            elif block.get("type") == "thinking":
                state.saw_reasoning = True
        elif event == "content_block_delta":
            delta = payload.get("delta") or {}
            if delta.get("type") == "text_delta":
                state.append_text(_text(delta.get("text")))
            elif delta.get("type") == "thinking_delta":
                state.saw_reasoning = True
        elif event == "message_delta":
            state.usage.update(payload.get("usage") or {})
            state.finish_reason = _finish((payload.get("delta") or {}).get("stop_reason"))
        elif event == "message_stop":
            if not state.finish_reason:
                raise ValueError("message_stop arrived without stop_reason")
            state.complete = True
    elif api == "generate-content":
        candidates = payload.get("candidates") or []
        if candidates:
            candidate = candidates[0]
            content, reasoning = _blocks_text((candidate.get("content") or {}).get("parts", []))
            state.append_text(content)
            state.saw_reasoning |= bool(reasoning)
            if candidate.get("finishReason"):
                state.finish_reason = _finish(candidate["finishReason"])
                state.complete = True
        if isinstance(payload.get("usageMetadata"), dict):
            state.usage.update(payload["usageMetadata"])
    state.usage_available = usage_details(api, state.usage)["usage_available"]


def decode_sse_data(data_lines: list[str]) -> tuple[str | None, dict[str, Any]]:
    data = "\n".join(data_lines)
    if data.strip() == "[DONE]":
        return "[DONE]", {}
    parsed = json.loads(data)
    if not isinstance(parsed, dict):
        raise ValueError("SSE payload must be a JSON object")
    return None, parsed


def iter_sse_events(lines: Iterable[str]) -> Iterable[tuple[str | None, dict[str, Any]]]:
    event_name: str | None = None
    data_lines: list[str] = []
    seen_data = False
    for raw in lines:
        line = raw.rstrip("\r\n")
        if not line:
            if data_lines:
                marker, parsed = decode_sse_data(data_lines)
                yield marker or event_name, parsed
                seen_data = True
            event_name, data_lines = None, []
        elif line.startswith("event:"):
            event_name = line[6:].strip() or None
        elif line.startswith("data:"):
            data_lines.append(line[5:].removeprefix(" "))
    if data_lines:
        raise ValueError("SSE stream ended before frame completion")
    if not seen_data:
        raise ValueError("SSE stream is empty")
