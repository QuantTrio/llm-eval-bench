from __future__ import annotations

import ast
import json
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
_MMMU_NUMBER = re.compile(r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?")
_MMMU_FINAL_ANSWER = re.compile(r"\bfinal[\s*_`]+answer\b[\s*_`]*(?::|：|is\b)?", re.I)


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
    if item.type == "mmmu_open":
        return (
            f"{item.question}\nReturn only your short final answer. "
            "For numerical answers, include a decimal value."
        )
    return item.question


def build_messages(item: DatasetItem) -> list[dict[str, object]]:
    if item.messages:
        return item.messages
    return [{"role": "user", "content": build_prompt(item)}]


def _contains_cjk(value: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", value))


def final_answer_view(output: str) -> str:
    """Return the answer portion while keeping the original output for artifacts."""
    lowered = output.casefold()
    closing = lowered.rfind("</think>")
    if closing >= 0:
        return output[closing + len("</think>") :].strip()
    if "<think>" in lowered:
        return ""
    return output.strip()


def parse_choice(output: str, valid_choices: set[str]) -> str | None:
    output = final_answer_view(output)
    for pattern in CHOICE_PATTERNS:
        matches = pattern.findall(output)
        if matches and matches[-1].upper() in valid_choices:
            return matches[-1].upper()
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
    output = final_answer_view(output)
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


def _normalize_mmmu(value: str) -> list[str | float]:
    value = value.strip()
    if not value:
        return []
    try:
        number = float(value.replace(",", ""))
    except ValueError:
        value = value.lower()
        return [f" {value}", f"{value} "] if len(value) == 1 else [value]
    return [round(number, 2)] if math.isfinite(number) else []


def parse_mmmu_open(output: str) -> list[str | float]:
    """MMMU-style candidate extraction, excluding hidden reasoning.

    Based on MMMU-Benchmark/MMMU's mmmu/utils/eval_utils.py (Apache-2.0).
    Unlike its overlapping number regexes, extract grouped decimals and scientific
    notation as complete numbers; prefer the last explicit final-answer section
    and retain trailing answer lines. Purely symbolic equivalence is not evaluated.
    """
    response = final_answer_view(output).strip().strip(".").lower()
    if not response:
        return []
    final_markers = list(_MMMU_FINAL_ANSWER.finditer(response))
    if final_markers:
        response = response[final_markers[-1].end() :].strip(" \t\r\n:*_`#")
        key_responses = [response] if response else []
    else:
        key_responses = _mmmu_key_responses(response)
    candidates = []
    for value in key_responses:
        for candidate in (value, *_MMMU_NUMBER.findall(value)):
            candidates.extend(_normalize_mmmu(candidate))
    return list(dict.fromkeys(candidates))


def _mmmu_key_responses(response: str) -> list[str]:
    parts = re.split(r"\.\s(?=[A-Z])|\n", response)
    indicators = ("could be ", "so ", "is ", "thus ", "therefore ", "final ", "answer ", "result ")
    key_responses = []
    for index, part in enumerate(parts):
        shortest = None
        for indicator in indicators + (("=",) if index == len(parts) - 1 else ()):
            if indicator in part:
                tail = part.rsplit(indicator, 1)[-1].strip()
                if not shortest or len(tail) < len(shortest):
                    shortest = tail
        if shortest and shortest not in {":", ",", ".", "!", "?", ";", "'"}:
            key_responses.append(shortest)
    key_responses = key_responses or [response]
    last_line = next((part.strip() for part in reversed(parts) if re.search(r"\w", part)), "")
    if last_line:
        key_responses.append(last_line)
    return key_responses


def _mmmu_references(answer: str) -> list[str | float]:
    # DatasetItem stores answers as strings, including upstream lists of aliases.
    alternatives = [answer]
    if answer.strip().startswith("["):
        try:
            value = ast.literal_eval(answer)
        except (ValueError, SyntaxError):
            value = None
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            alternatives = value
    return [normalized for value in alternatives for normalized in _normalize_mmmu(value)]


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
        parsed = final_answer_view(output)
        references = [item.answer, *(item.metadata.get("accepted_answers") or [])]
        score = max(token_f1(parsed, reference) for reference in references if reference)
        return parsed, score, not bool(parsed)
    if item.type == "mmmu_open":
        candidates = parse_mmmu_open(output)
        references = _mmmu_references(item.answer)
        correct = any(
            reference in candidate
            if isinstance(reference, str) and isinstance(candidate, str)
            else reference == candidate
            for candidate in candidates
            for reference in references
        )
        parsed = json.dumps(candidates, ensure_ascii=False) if candidates else None
        return parsed, float(correct), not bool(candidates)
    parsed = normalize_text(final_answer_view(output))
    gold = normalize_text(item.answer)
    return parsed, float(parsed == gold), not bool(parsed)
