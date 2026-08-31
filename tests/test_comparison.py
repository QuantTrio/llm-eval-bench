from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from llmbench.cli import app
from llmbench.comparison import (
    IncomparableRunsError,
    compare_run_directories,
    evaluate_policy,
)
from llmbench.metrics import summarize
from llmbench.report import write_run_artifacts
from llmbench.schemas import RequestResult


def row(question_id: str, score: float, *, model: str) -> RequestResult:
    return RequestResult(
        run_id=model,
        model=model,
        dataset="test",
        benchmark_category="Test",
        question_type="multiple_choice",
        metric="accuracy",
        question_id=question_id,
        sample_id=1,
        concurrency=1,
        prompt="prompt",
        raw_output="A",
        parsed_answer="A" if score else "B",
        gold_answer="A",
        score=score,
        parse_failed=False,
        latency_ms=100,
        ttft_ms=10,
        tpot_ms=2,
        input_tokens=5,
        output_tokens=2,
        finish_reason="stop",
        error=None,
        error_type=None,
        http_status=None,
        attempts=1,
        max_tokens=4096,
    )


def write_run(directory, rows: list[RequestResult], *, model: str, seed: int = 42) -> None:
    config = {
        "datasets": ["test"],
        "concurrency": 1,
        "temperature": 0,
        "seed": seed,
        "n_samples": 1,
    }
    summary = summarize(
        rows,
        run_id=model,
        mode="run",
        model=model,
        base_url=f"http://{model}/v1",
        elapsed_seconds=1,
        config=config,
    )
    write_run_artifacts(directory, summary, rows)
    manifest = {
        "schema_version": 2,
        "run_id": model,
        "mode": "run",
        "model": model,
        "base_url": f"http://{model}/v1",
        "config": config,
        "datasets": {"test": {"sha256": "same"}},
        "question_keys_sha256": "same-keys",
        "request_count": len(rows),
    }
    (directory / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_paired_comparison_bootstrap_and_policy(tmp_path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    write_run(baseline, [row("q1", 1, model="base"), row("q2", 1, model="base")], model="base")
    write_run(candidate, [row("q1", 1, model="cand"), row("q2", 0, model="cand")], model="cand")
    comparison = compare_run_directories(baseline, candidate, bootstrap_samples=500)
    assert comparison["quality"]["sample_mean_change"] == -0.5
    assert comparison["quality"]["transitions"]["correct_to_wrong"] == 1
    assert comparison["quality"]["by_dataset"]["test"]["absolute_change"] == -0.5
    assert comparison["quality"]["confidence_interval"]["upper"] <= 0
    policy = evaluate_policy(
        comparison,
        {"gates": {"max_score_drop": 0.1, "max_dataset_drop": 0.2}},
    )
    assert policy["passed"] is False


def test_incomparable_manifest_and_cli_exit_codes(tmp_path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    write_run(baseline, [row("q1", 1, model="base")], model="base")
    write_run(candidate, [row("q1", 0, model="cand")], model="cand", seed=99)
    with pytest.raises(IncomparableRunsError, match="config"):
        compare_run_directories(baseline, candidate, bootstrap_samples=100)

    result = CliRunner().invoke(
        app,
        ["compare", "--baseline", str(baseline), "--candidate", str(candidate)],
    )
    assert result.exit_code == 3


def test_cli_policy_failure_is_exit_two(tmp_path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    write_run(baseline, [row("q1", 1, model="base")], model="base")
    write_run(candidate, [row("q1", 0, model="cand")], model="cand")
    policy = tmp_path / "policy.yaml"
    policy.write_text("gates:\n  max_score_drop: 0.1\n", encoding="utf-8")
    result = CliRunner().invoke(
        app,
        [
            "compare",
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--policy",
            str(policy),
            "--bootstrap-samples",
            "100",
            "--report",
            str(tmp_path / "comparison.html"),
        ],
    )
    assert result.exit_code == 2, result.exception
    assert (tmp_path / "paired_results.jsonl").exists()
