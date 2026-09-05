from __future__ import annotations

import json

import pytest

from llmbench.metrics import summarize
from llmbench.report import compare_summaries, write_comparison, write_run_artifacts
from llmbench.schemas import RequestResult


def result(
    question_id: str,
    score: float,
    *,
    dataset: str = "mmlu-pro",
    category: str = "Comprehensive",
    question_type: str = "multiple_choice",
    metric: str = "accuracy",
    sample_id: int = 1,
) -> RequestResult:
    return RequestResult(
        run_id="run-1",
        model="model-a",
        dataset=dataset,
        benchmark_category=category,
        question_type=question_type,
        metric=metric,
        question_id=question_id,
        sample_id=sample_id,
        concurrency=2,
        prompt="prompt",
        raw_output="A",
        parsed_answer="A",
        gold_answer="A" if score else "B",
        score=score,
        parse_failed=False,
        latency_ms=100 + sample_id,
        ttft_ms=20,
        tpot_ms=5,
        input_tokens=10,
        output_tokens=5,
        finish_reason="stop",
        error=None,
        error_type=None,
        http_status=None,
        attempts=1,
        usage_available=True,
    )


def make_summary(rows: list[RequestResult], run_id: str = "run-1") -> dict:
    return summarize(
        rows,
        run_id=run_id,
        mode="run",
        model="model-a",
        base_url="http://localhost/v1",
        elapsed_seconds=1.0,
        config={"datasets": sorted({row.dataset for row in rows}), "concurrency": 2},
    )


def test_category_type_and_dataset_breakdowns(tmp_path) -> None:
    rows = [
        result("q1", 1),
        result("q2", 0),
        result(
            "d1",
            0.8,
            dataset="drop",
            category="Reading Comprehension",
            question_type="f1",
            metric="token_f1",
        ),
    ]
    summary = make_summary(rows)
    assert summary["quality"]["sample_mean_score"] == pytest.approx(0.6)
    assert summary["quality"]["by_dataset"]["drop"]["metric"] == "token_f1"
    assert "accuracy" not in summary["quality"]["by_dataset"]["drop"]
    assert summary["quality"]["by_category"]["Comprehensive"]["macro_mean_score"] == 0.5
    assert summary["quality"]["by_question_type"]["f1"]["score"] == 0.8
    assert "pass_at_k" not in summary["quality"]
    paths = write_run_artifacts(tmp_path / "run", summary, rows)
    assert all(path.exists() for path in paths.values())
    assert "Category breakdown" in paths["markdown"].read_text(encoding="utf-8")
    assert len(paths["raw"].read_text(encoding="utf-8").splitlines()) == 3
    html = paths["html"].read_text(encoding="utf-8")
    assert "Category radar" in html
    assert "aria-label='Category score radar'" in html
    assert "Dataset scores" in html
    assert "Question details" in html
    assert "prompt" in html


def test_standard_pass_at_k_only_for_repeated_binary_samples() -> None:
    rows = [
        result("q1", 0, sample_id=1),
        result("q1", 1, sample_id=2),
        result("q2", 0, sample_id=1),
        result("q2", 0, sample_id=2),
        result("f1", 0.2, dataset="drop", question_type="f1", metric="token_f1", sample_id=1),
        result("f1", 0.3, dataset="drop", question_type="f1", metric="token_f1", sample_id=2),
    ]
    summary = make_summary(rows)
    assert summary["quality"]["pass_k"] == 2
    assert summary["quality"]["pass_at_k"] == 0.5
    assert summary["quality"]["accuracy_at_1"] == 0.0


def test_truncation_marks_dataset_quality_invalid() -> None:
    rows = [result(f"q{index}", 1) for index in range(20)]
    rows[0].finish_reason = "length"
    rows[1].finish_reason = "length"
    summary = make_summary(rows)
    dataset = summary["quality"]["by_dataset"]["mmlu-pro"]
    assert dataset["truncation_rate"] == 0.1
    assert dataset["quality_valid"] is False
    assert summary["quality"]["quality_valid"] is False
    assert summary["quality"]["invalid_datasets"] == ["mmlu-pro"]


def test_comparison_artifacts(tmp_path) -> None:
    baseline = make_summary([result("q1", 1), result("q2", 1)], "baseline")
    candidate = make_summary([result("q1", 1), result("q2", 0)], "candidate")
    baseline["config"]["memory_gb"] = 80.0
    candidate["config"]["memory_gb"] = 24.0
    comparison = compare_summaries(baseline, candidate)
    assert comparison["quality"]["absolute_change"] == -0.5
    assert comparison["performance"]["memory_reduction"] == pytest.approx(0.7)
    paths = write_comparison(tmp_path / "compare.html", comparison)
    assert all(path.exists() for path in paths.values())
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["candidate"]["run_id"] == "candidate"
    assert "Memory reduction: 70.00%" in paths["markdown"].read_text(encoding="utf-8")


def test_missing_usage_is_unknown_not_zero_throughput(tmp_path) -> None:
    rows = [result("q1", 1), result("q2", 1)]
    rows[1].usage_available = False
    summary = make_summary(rows)
    performance = summary["performance"]
    assert performance["usage_reported_requests"] == 1
    assert performance["usage_complete"] is False
    assert performance["output_tokens_per_second"] is None
    assert performance["input_tokens_per_second"] is None
    assert performance["output_tokens"] is None
    paths = write_run_artifacts(tmp_path, summary, rows)
    assert "Output tokens/s: n/a" in paths["markdown"].read_text()


def test_usage_includes_reported_reasoning_and_cached_counts() -> None:
    rows = [result("q1", 1), result("q2", 1)]
    for item in rows:
        item.reasoning_tokens = 2
        item.cached_input_tokens = 4
    performance = make_summary(rows)["performance"]
    assert performance["usage_complete"] is True
    assert performance["output_tokens"] == 10
    assert performance["output_tokens_per_second"] == 10
    assert performance["reasoning_tokens"] == 4
    assert performance["cached_input_tokens"] == 8


def test_http_errors_and_missing_scores_cannot_claim_valid_quality() -> None:
    item = result("q1", 0)
    item.error = "service unavailable"
    item.error_type = "http_503"
    quality = make_summary([item])["quality"]
    assert quality["quality_valid"] is False
    assert quality["by_dataset"]["mmlu-pro"]["failed_samples"] == 1
    assert make_summary([])["quality"]["quality_valid"] is False


def test_parse_failure_threshold_invalidates_score() -> None:
    rows = [result(f"q{i}", 0) for i in range(10)]
    rows[0].parse_failed = True
    assert make_summary(rows)["quality"]["quality_valid"] is False
