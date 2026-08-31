from __future__ import annotations

import base64
import hashlib
import json
import math
import mimetypes
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from .client import OpenAICompatibleClient
from .schemas import DatasetItem


def _asset_bytes(item: DatasetItem, asset: str) -> bytes:
    package = item.metadata.get("resource_package")
    if package:
        return resources.files(str(package)).joinpath(asset).read_bytes()
    return Path(asset).expanduser().read_bytes()


def asset_data_url(item: DatasetItem, asset: str) -> str:
    mime = mimetypes.guess_type(asset)[0] or "application/octet-stream"
    encoded = base64.b64encode(_asset_bytes(item, asset)).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def decrypt_browsecomp(ciphertext: str, canary: str) -> str:
    encrypted = base64.b64decode(ciphertext)
    digest = hashlib.sha256(canary.encode()).digest()
    key = (digest * (len(encrypted) // len(digest) + 1))[: len(encrypted)]
    return bytes(left ^ right for left, right in zip(encrypted, key, strict=True)).decode()


def multimodal_messages(item: DatasetItem) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": item.question}]
    for asset in item.metadata.get("assets") or []:
        mime = mimetypes.guess_type(str(asset))[0] or "application/octet-stream"
        if not mime.startswith("image/"):
            raise ValueError(f"unsupported multimodal asset type: {mime}")
        content.append(
            {"type": "image_url", "image_url": {"url": asset_data_url(item, str(asset))}}
        )
    return [{"role": "user", "content": content}]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("embedding vectors must have the same non-zero dimension")
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(
        sum(value * value for value in right)
    )
    if denominator == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / denominator


async def score_embedding_item(
    client: OpenAICompatibleClient,
    *,
    model: str,
    item: DatasetItem,
) -> tuple[float | None, dict[str, Any]]:
    positive = str(item.metadata.get("positive") or item.answer or "")
    negatives = [str(value) for value in item.metadata.get("negatives") or []]
    if not positive or not negatives:
        raise ValueError("embedding item requires one positive and at least one negative")
    inputs = [item.question, positive, *negatives]
    result = await client.embed(model=model, inputs=inputs)
    if result.error:
        return None, {"error": result.error, "error_type": result.error_type}
    query, positive_vector, *negative_vectors = result.vectors
    positive_score = cosine_similarity(query, positive_vector)
    negative_scores = [cosine_similarity(query, vector) for vector in negative_vectors]
    return float(positive_score > max(negative_scores)), {
        "positive_similarity": positive_score,
        "max_negative_similarity": max(negative_scores),
        "latency_ms": result.latency_ms,
        "input_tokens": result.input_tokens,
    }


def strip_code_fences(value: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", value, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else value.strip()


def humaneval_executor_payload(
    item: DatasetItem,
    candidate_output: str,
    *,
    image: str,
) -> dict[str, Any]:
    entry_point = item.metadata.get("entry_point")
    test = item.metadata.get("test")
    if not entry_point or not test:
        raise ValueError("HumanEval item is missing entry_point or test")
    completion = strip_code_fences(candidate_output)
    program = completion if f"def {entry_point}" in completion else item.question + completion
    script = f"{program}\n\n{test}\n\ncheck({entry_point})\n"
    return {"image": image, "command": ["-c", script], "network": False}


@dataclass(slots=True)
class JudgeResult:
    score: float | None
    scores: list[float]
    consistency: float | None
    outputs: list[str]
    errors: list[str]


def parse_judge_score(output: str) -> float | None:
    try:
        payload = json.loads(output)
        score = float(payload["score"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        match = re.search(r'"?score"?\s*[:=]\s*([01](?:\.\d+)?)', output, re.IGNORECASE)
        if not match:
            return None
        score = float(match.group(1))
    return min(1.0, max(0.0, score))


async def judge_response(
    client: OpenAICompatibleClient,
    *,
    model: str,
    item: DatasetItem,
    candidate_output: str,
    repeats: int = 3,
    max_tokens: int = 1024,
) -> JudgeResult:
    rubric = str(item.metadata.get("rubric") or "Score correctness and instruction adherence.")
    prompt = (
        f"Question:\n{item.question}\n\nCandidate response:\n{candidate_output}\n\n"
        f'Rubric:\n{rubric}\n\nReturn JSON {{"score": <number from 0 to 1>}}.'
    )
    scores = []
    outputs = []
    errors = []
    for sample_id in range(repeats):
        result = await client.complete(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            top_p=1,
            max_tokens=max_tokens,
            stream=False,
            seed=sample_id + 1,
        )
        outputs.append(result.content)
        if result.error:
            errors.append(result.error)
            continue
        score = parse_judge_score(result.content)
        if score is None:
            errors.append("judge_parse_failed")
        else:
            scores.append(score)
    return JudgeResult(
        score=mean(scores) if scores else None,
        scores=scores,
        consistency=(1.0 - min(1.0, pstdev(scores))) if len(scores) > 1 else None,
        outputs=outputs,
        errors=errors,
    )
