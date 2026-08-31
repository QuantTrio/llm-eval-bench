from __future__ import annotations

import base64
import hashlib
import json

import httpx
import pytest

from llmbench.adapters import (
    asset_data_url,
    cosine_similarity,
    decrypt_browsecomp,
    humaneval_executor_payload,
    judge_response,
    multimodal_messages,
    parse_judge_score,
    score_embedding_item,
    strip_code_fences,
)
from llmbench.client import OpenAICompatibleClient
from llmbench.schemas import DatasetItem


def test_multimodal_assets_and_humaneval_payload(tmp_path) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"png")
    item = DatasetItem(
        id="mm",
        dataset="mmmu",
        type="multiple_choice",
        question="What is shown?",
        answer="A",
        metadata={"assets": [str(image)]},
    )
    assert asset_data_url(item, str(image)).startswith("data:image/png;base64,")
    messages = multimodal_messages(item)
    assert messages[0]["content"][1]["type"] == "image_url"

    code = DatasetItem(
        id="HumanEval/0",
        dataset="humaneval",
        type="code",
        question="def answer():\n",
        answer="    return 42",
        metadata={"entry_point": "answer", "test": "def check(fn): assert fn() == 42"},
    )
    payload = humaneval_executor_payload(
        code,
        "```python\n    return 42\n```",
        image="sandbox:test",
    )
    assert payload["network"] is False
    assert "check(answer)" in payload["command"][1]
    assert strip_code_fences("```python\nprint(1)\n```") == "print(1)"
    document = tmp_path / "document.pdf"
    document.write_bytes(b"pdf")
    item.metadata["assets"] = [str(document)]
    with pytest.raises(ValueError, match="unsupported multimodal"):
        multimodal_messages(item)
    with pytest.raises(ValueError, match="same non-zero"):
        cosine_similarity([1], [1, 2])
    assert cosine_similarity([0, 0], [1, 1]) == 0
    assert parse_judge_score('{"score": 2}') == 1
    assert parse_judge_score("invalid") is None
    plaintext = b"sensitive question"
    canary = "canary"
    digest = hashlib.sha256(canary.encode()).digest()
    ciphertext = base64.b64encode(
        bytes(value ^ digest[index % len(digest)] for index, value in enumerate(plaintext))
    ).decode()
    assert decrypt_browsecomp(ciphertext, canary) == plaintext.decode()


@pytest.mark.asyncio
async def test_embedding_and_judge_adapters() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if request.url.path.endswith("/embeddings"):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"index": 0, "embedding": [1, 0]},
                        {"index": 1, "embedding": [1, 0]},
                        {"index": 2, "embedding": [0, 1]},
                    ],
                    "usage": {"prompt_tokens": 10},
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": json.dumps({"score": 0.8})}, "finish_reason": "stop"}
                ]
            },
        )

    item = DatasetItem(
        id="embed",
        dataset="mteb",
        type="embedding",
        question="query",
        metadata={"positive": "right", "negatives": ["wrong"]},
    )
    async with OpenAICompatibleClient(
        base_url="http://test/v1",
        api_key="EMPTY",
        transport=httpx.MockTransport(handler),
    ) as client:
        score, details = await score_embedding_item(client, model="embed", item=item)
        assert score == 1
        assert details["positive_similarity"] == 1
        judged = await judge_response(
            client,
            model="judge",
            item=item,
            candidate_output="answer",
            repeats=3,
        )
    assert judged.score == pytest.approx(0.8)
    assert judged.consistency == 1
    assert calls == 4
    assert cosine_similarity([1, 0], [1, 0]) == 1
    assert parse_judge_score("score: 0.4") == 0.4
