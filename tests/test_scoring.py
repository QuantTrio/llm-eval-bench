from __future__ import annotations

import pytest

from llmbench.schemas import DatasetItem
from llmbench.scoring import (
    build_prompt,
    parse_choice,
    parse_math,
    parse_mmmu_open,
    score_output,
    token_f1,
)


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


def test_scoring_uses_answer_after_thinking_block() -> None:
    item = DatasetItem(
        id="q1",
        dataset="test",
        type="multiple_choice",
        question="Which one?",
        choices={"A": "first", "B": "second"},
        answer="B",
    )
    assert score_output(item, '<think>I considered A.</think>{"answer":"B"}') == (
        "B",
        1.0,
        False,
    )
    assert score_output(item, "<think>I considered B but was truncated") == (None, 0.0, True)


# The 24 open-answer references in the bundled 500-question MMMU selection.
@pytest.mark.parametrize(
    "answer",
    [
        "2.83",
        "242110.62",
        "2",
        "trans-1-Chloro-4-methylcyclohexane",
        "8.4",
        "['Tampa', 'Florida']",
        "62.6",
        "71.6",
        "360",
        "60",
        "C",
        "1/64",
        "2000000",
        "0.3",
        "0.10",
        "10",
        "12.97",
        "1.06",
        "Transformation",
        "8200",
        "['24/7', '3.429']",
        "20",
        "1000",
        "65",
    ],
)
def test_mmmu_all_bundled_open_references_accept_self_and_reject_wrong(answer: str) -> None:
    item = DatasetItem(id="open", dataset="mmmu", type="mmmu_open", question="?", answer=answer)
    parsed, score, failed = score_output(item, answer)
    assert parsed
    assert (score, failed) == (1.0, False)
    assert score_output(item, "unrelated_response")[1:] == (0.0, False)


@pytest.mark.parametrize(
    ("answer", "output", "expected"),
    [
        ("2.83", "The final answer is 2.834 A.", 1.0),
        ("2.83", "2.84", 0.0),
        ("242110.62", "Final answer: $242,110.62", 1.0),
        ("-1250", "The result is -1.25e3 V.", 1.0),
        ("2000000", "2E+6 dollars", 1.0),
        ("0.10", "The answer is .1", 1.0),
        ("0.3", "x = 0.3", 1.0),
        ("Transformation", "The process is TRANSFORMATION.", 1.0),
        ("C", "The answer is C.", 1.0),
        ("C", "Carbon", 0.0),
        ("1/64", "Probability: 1/64", 1.0),
        ("1/64", "0.015625", 0.0),
        ("['Tampa', 'Florida']", "Tampa", 1.0),
        ('["Tampa", "Florida"]', "Florida", 1.0),
        ("['Tampa', 'Florida']", "Boston", 0.0),
        ("['24/7', '3.429']", "24/7 feet per second", 1.0),
        ("['24/7', '3.429']", "3.43", 1.0),
        ("['24/7', '3.429']", "3.42", 0.0),
        ("10", "20\nThe final answer is 10", 1.0),
        ("10", "10\nThe final answer is 20", 0.0),
        ("2.83", "<think>2.83</think>9", 0.0),
        ("2.83", "<think>9</think>The answer is 2.83", 1.0),
        ("3", "1.25e3", 0.0),
        ("110.62", "242,110.62", 0.0),
    ],
)
def test_mmmu_open_formatted_answers(answer: str, output: str, expected: float) -> None:
    item = DatasetItem(id="open", dataset="mmmu", type="mmmu_open", question="?", answer=answer)
    assert score_output(item, output)[1:] == (expected, False)


@pytest.mark.parametrize("output", ["", "   ", "...", "<think>2.83", "<think>2.83</think>"])
def test_mmmu_open_empty_or_hidden_answers_fail_parsing(output: str) -> None:
    item = DatasetItem(id="open", dataset="mmmu", type="mmmu_open", question="?", answer="2.83")
    assert score_output(item, output) == (None, 0.0, True)


def test_mmmu_open_prompt_and_deterministic_candidates() -> None:
    item = DatasetItem(
        id="open", dataset="mmmu", type="mmmu_open", question="Read <image 1>.", answer="2.83"
    )
    assert build_prompt(item) == (
        "Read <image 1>.\nReturn only your short final answer. "
        "For numerical answers, include a decimal value."
    )
    assert parse_mmmu_open("2.83\n2.83") == ["2.83\n2.83", 2.83]


def test_mmmu_rules_do_not_change_generic_exact_match() -> None:
    item = DatasetItem(
        id="exact", dataset="custom", type="exact_match", question="?", answer="Tampa"
    )
    assert score_output(item, "It is Tampa") == ("it is tampa", 0.0, False)


@pytest.mark.parametrize(
    "output",
    [
        # Sanitized Responses JSON capture: the unmarked final line was discarded.
        r"""The waveform in (a) is a sine wave with a peak amplitude of \(A = 4\).
The effective (rms) value of a sinusoidal waveform is given by \(\frac{I_{max}}{\sqrt{2}}\).
Therefore, the rms value is \(\frac{4}{\sqrt{2}} = 2\sqrt{2}\).

\(2\sqrt{2}\) (or \(\approx 2.83\))""",
        # Sanitized Responses SSE capture: the Markdown heading displaced its answer.
        r"""The peak amplitude is given as \(A = 4\) A.
The effective value is \(2\sqrt{2}\) A.

**Final Answer:**
\(2\sqrt{2}\) A (or approximately 2.83 A)""",
    ],
)
def test_mmmu_open_retains_decimal_from_live_responses(output: str) -> None:
    item = DatasetItem(id="open", dataset="mmmu", type="mmmu_open", question="?", answer="2.83")
    assert score_output(item, output)[1:] == (1.0, False)


@pytest.mark.parametrize(
    "output",
    [
        "The result is 4.\nFinal answer: 2.83",
        "Final answer: 4\n**Final Answer:**\n2.83",
        "The result is 4.\n### Final **Answer**\n\n2.83",
        "<think>Final answer: 4</think>Final answer is 2.83",
    ],
)
def test_mmmu_open_last_final_answer_excludes_earlier_values(output: str) -> None:
    item = DatasetItem(id="open", dataset="mmmu", type="mmmu_open", question="?", answer="2.83")
    assert score_output(item, output)[1:] == (1.0, False)
    item.answer = "4"
    assert score_output(item, output)[1:] == (0.0, False)
    assert 4.0 not in parse_mmmu_open(output)


def test_mmmu_open_does_not_evaluate_symbolic_equivalence() -> None:
    item = DatasetItem(id="open", dataset="mmmu", type="mmmu_open", question="?", answer="2.83")
    assert score_output(item, r"Final answer: \(2\sqrt{2}\)")[1:] == (0.0, False)
    assert score_output(item, "The result is 2.83\n**Final Answer:**") == (None, 0.0, True)
