from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class DatasetItem:
    id: str
    dataset: str
    type: str
    question: str
    answer: str | None = None
    choices: dict[str, str] = field(default_factory=dict)
    subset: str | None = None
    messages: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_image(self) -> bool:
        if self.metadata.get("capability", "chat") not in {"chat", "multimodal"}:
            return False
        if self.metadata.get("capability") == "multimodal" or self.metadata.get("assets"):
            return True
        return any(
            isinstance(block, dict) and block.get("type") in {"image_url", "image", "input_image"}
            for message in self.messages or []
            for block in (
                message.get("content", []) if isinstance(message.get("content"), list) else []
            )
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any], *, source: str = "custom") -> DatasetItem:
        required = ("id", "type", "question")
        missing = [key for key in required if not value.get(key)]
        if missing:
            raise ValueError(f"Dataset record is missing required fields {missing}: {source}")
        choices = value.get("choices") or {}
        if isinstance(choices, list):
            choices = {chr(65 + index): text for index, text in enumerate(choices)}
        return cls(
            id=str(value["id"]),
            dataset=str(value.get("dataset") or source),
            type=str(value["type"]),
            question=str(value["question"]),
            answer=None if value.get("answer") is None else str(value["answer"]),
            choices={str(key): str(text) for key, text in choices.items()},
            subset=None if value.get("subset") is None else str(value["subset"]),
            messages=value.get("messages"),
            metadata=dict(value.get("metadata") or {}),
        )


@dataclass(slots=True)
class CompletionResult:
    content: str = ""
    latency_ms: float = 0.0
    ttft_ms: float | None = None
    tpot_ms: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int | None = None
    cached_input_tokens: int | None = None
    usage_available: bool = False
    finish_reason: str | None = None
    request_id: str | None = None
    raw_response: dict[str, Any] | None = None
    error: str | None = None
    error_type: str | None = None
    http_status: int | None = None
    attempts: int = 1
    attempt_latency_ms: float | None = None


@dataclass(slots=True)
class EmbeddingResult:
    vectors: list[list[float]] = field(default_factory=list)
    latency_ms: float = 0.0
    input_tokens: int = 0
    error: str | None = None
    error_type: str | None = None


@dataclass(slots=True)
class RequestResult:
    run_id: str
    model: str
    dataset: str
    benchmark_category: str
    question_type: str
    metric: str
    question_id: str
    sample_id: int
    concurrency: int
    prompt: str
    raw_output: str
    parsed_answer: str | None
    gold_answer: str | None
    score: float | None
    parse_failed: bool
    latency_ms: float
    ttft_ms: float | None
    tpot_ms: float | None
    input_tokens: int
    output_tokens: int
    error: str | None
    error_type: str | None
    http_status: int | None
    attempts: int
    reasoning_tokens: int | None = None
    cached_input_tokens: int | None = None
    usage_available: bool = False
    finish_reason: str | None = None
    request_id: str | None = None
    raw_response: dict[str, Any] | None = None
    max_tokens: int | None = None
    attempt_latency_ms: float | None = None
    images: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.dataset, self.question_id, self.sample_id)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RequestResult:
        fields = cls.__dataclass_fields__
        return cls(**{key: value[key] for key in fields if key in value})
