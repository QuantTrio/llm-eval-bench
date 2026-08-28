from __future__ import annotations

import math
import re
import string
from collections import Counter
from decimal import Decimal, InvalidOperation

from .schemas import DatasetItem

CHOICE_PATTERNS = (
    re.compile(r'["\']answer["\']\s*:\s*["\']([A-Z])["\']', re.I),
    re.compile(r"(?:final\s+answer|answer)\s*(?:is|:|：)?\s*\(?([A-Z])\)?", re.I),
    re.compile(r"答案\s*(?:是|为|:|：)?\s*\(?([A-Z])\)?", re.I),
    re.compile(r"(?:选|选择)\s*\(?([A-Z])\)?", re.I),
    re.compile(r"\(([A-Z])\)", re.I),
    re.compile(r"^\s*([A-Z])(?:[.、:：\s]|$)", re.I),
)


def build_prompt(item: DatasetItem) -> str:
    if item.type == "multiple_choice":
        options = "\n".join(f"{key}. {text}" for key, text in item.choices.items())
        language_hint = (
            '请以 JSON 格式回答，例如：{"answer":"C"}。'
            if _contains_cjk(item.question)
            else 'Return JSON with only the answer field, for example: {"answer":"C"}.'
        )
        return f"{item.question}\n{options}\n{language_hint}"
    if item.type == "math":
        return (
            f"{item.question}\nPlease reason step by step, and put your final answer within "
            "\\boxed{<answer>}."
        )
    if item.type == "f1":
        return f"{item.question}\nRespond with only the shortest answer supported by the passage."
    return item.question


def build_messages(item: DatasetItem) -> list[dict[str, str]]:
    if item.messages:
        return item.messages
    return [{"role": "user", "content": build_prompt(item)}]


def _contains_cjk(value: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", value))


def parse_choice(output: str, valid_choices: set[str]) -> str | None:
    for pattern in CHOICE_PATTERNS:
        match = pattern.search(output.strip())
        if match and match.group(1).upper() in valid_choices:
            return match.group(1).upper()
    return None


def _normalize_number(value: str) -> str | None:
    cleaned = value.strip().replace(",", "").replace("−", "-")
    cleaned = re.sub(r"(?<=\d)%$", "", cleaned)
    try:
        number = Decimal(cleaned)
    except InvalidOperation:
        return None
    if not number.is_finite():
        return None
    normalized = format(number.normalize(), "f")
    return "0" if normalized in {"-0", ""} else normalized


def parse_math(output: str) -> str | None:
    patterns = (
        re.compile(r"\\boxed\{\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*\}"),
        re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)"),
        re.compile(
            r"(?:final\s+answer|answer|答案)\s*(?:is|:|：|是|为)?\s*([-+]?\d[\d,]*(?:\.\d+)?)", re.I
        ),
    )
    for pattern in patterns:
        matches = pattern.findall(output)
        if matches:
            return _normalize_number(matches[-1])
    numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", output)
    return _normalize_number(numbers[-1]) if numbers else None


def normalize_text(value: str) -> str:
    value = value.casefold().strip()
    value = re.sub(r"[^\w\s]", "", value)
    return re.sub(r"\s+", " ", value)


def token_f1(prediction: str, reference: str) -> float:
    def tokens(value: str) -> list[str]:
        lowered = value.casefold()
        lowered = lowered.translate(str.maketrans("", "", string.punctuation))
        lowered = re.sub(r"\b(a|an|the)\b", " ", lowered)
        return re.sub(r"\s+", " ", lowered).strip().split()

    predicted = tokens(prediction)
    gold = tokens(reference)
    if not predicted or not gold:
        return float(predicted == gold)
    overlap = sum((Counter(predicted) & Counter(gold)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def score_output(item: DatasetItem, output: str) -> tuple[str | None, float | None, bool]:
    if item.answer is None or item.type in {"stress", "chat"}:
        return None, None, False
    if item.type == "multiple_choice":
        parsed = parse_choice(output, set(item.choices))
        return parsed, float(parsed == item.answer.upper()) if parsed else 0.0, parsed is None
    if item.type == "math":
        parsed = parse_math(output)
        gold = parse_math(item.answer) or _normalize_number(item.answer)
        if parsed is None or gold is None:
            return parsed, 0.0, True
        try:
            correct = math.isclose(float(parsed), float(gold), rel_tol=1e-9, abs_tol=1e-9)
        except ValueError:
            correct = parsed == gold
        return parsed, float(correct), False
    if item.type == "f1":
        parsed = output.strip()
        references = [item.answer, *(item.metadata.get("accepted_answers") or [])]
        score = max(token_f1(parsed, reference) for reference in references if reference)
        return parsed, score, not bool(parsed)
    parsed = normalize_text(output)
    gold = normalize_text(item.answer)
    return parsed, float(parsed == gold), not bool(parsed)
