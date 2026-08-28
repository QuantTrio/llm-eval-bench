from __future__ import annotations

import pytest

from llmbench.schemas import DatasetItem
from llmbench.scoring import build_prompt, parse_choice, parse_math, score_output, token_f1


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ('{"answer":"C"}', "C"),
        ("The answer is C.", "C"),
        ("答案是 C", "C"),
        ("I choose (C).", "C"),
        ("C", "C"),
        ("Option Z", None),
    ],
)
def test_parse_choice_formats(output: str, expected: str | None) -> None:
    assert parse_choice(output, {"A", "B", "C", "D"}) == expected


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        (r"Therefore, \boxed{1,024}.", "1024"),
        ("#### 42.0", "42"),
        ("The final answer is -3.50", "-3.5"),
    ],
)
def test_parse_math_formats(output: str, expected: str) -> None:
    assert parse_math(output) == expected


def test_multiple_choice_prompt_uses_json_contract() -> None:
    item = DatasetItem(
        id="q1",
        dataset="test",
        type="multiple_choice",
        question="Which one?",
        choices={"A": "first", "B": "second"},
        answer="B",
    )
    prompt = build_prompt(item)
    assert '{"answer":"C"}' in prompt
    assert "A. first" in prompt
    assert score_output(item, '{"answer":"B"}') == ("B", 1.0, False)


def test_math_and_f1_scoring() -> None:
    math_item = DatasetItem(id="m1", dataset="math", type="math", question="1+1?", answer="#### 2")
    assert score_output(math_item, r"Reasoning. \boxed{2}") == ("2", 1.0, False)

    f1_item = DatasetItem(
        id="d1",
        dataset="drop",
        type="f1",
        question="Passage... Answer:",
        answer="the red fox",
    )
    parsed, score, failed = score_output(f1_item, "red fox")
    assert parsed == "red fox"
    assert score == 1.0
    assert failed is False
    assert token_f1("red fox jumped", "the red fox") == pytest.approx(0.8)
