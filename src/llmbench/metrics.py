from __future__ import annotations

from collections import Counter, defaultdict
from math import comb
from statistics import mean
from typing import Any

from .schemas import RequestResult

BINARY_METRICS = {"accuracy", "exact_match", "mc1_accuracy"}


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _distribution(values: list[float]) -> dict[str, float | None]:
    return {
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


def summarize(
    results: list[RequestResult],
    *,
    run_id: str,
    mode: str,
    model: str,
    base_url: str,
    elapsed_seconds: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    total = len(results)
    unsupported = [result for result in results if result.error_type == "unsupported_capability"]
    supported = [result for result in results if result.error_type != "unsupported_capability"]
    successful = [result for result in supported if result.error is None]
    scored = [result for result in results if result.score is not None]
    latency = [result.latency_ms for result in successful]
    ttft = [result.ttft_ms for result in successful if result.ttft_ms is not None]
    tpot = [result.tpot_ms for result in successful if result.tpot_ms is not None]
    attempt_latency = [
        result.attempt_latency_ms for result in successful if result.attempt_latency_ms is not None
    ]
    errors = Counter(result.error_type or "unknown" for result in supported if result.error)

    quality: dict[str, Any] = {
        "scored_samples": len(scored),
        "sample_mean_score": mean(result.score or 0.0 for result in scored) if scored else None,
        "composite_score": None,
        "parse_failed": sum(result.parse_failed for result in scored),
        "parse_fail_rate": (
            sum(result.parse_failed for result in scored) / len(scored) if scored else None
        ),
        "by_dataset": {},
        "by_category": {},
        "by_question_type": {},
        "by_metric": {},
    }
    grouped_dataset: dict[str, list[RequestResult]] = defaultdict(list)
    grouped_category: dict[str, list[RequestResult]] = defaultdict(list)
    grouped_type: dict[str, list[RequestResult]] = defaultdict(list)
    grouped_question: dict[tuple[str, str], list[RequestResult]] = defaultdict(list)
    grouped_metric: dict[str, list[RequestResult]] = defaultdict(list)
    for result in scored:
        grouped_dataset[result.dataset].append(result)
        grouped_category[result.benchmark_category].append(result)
        grouped_type[result.question_type].append(result)
        grouped_question[(result.dataset, result.question_id)].append(result)
        grouped_metric[result.metric].append(result)
    for dataset, rows in sorted(grouped_dataset.items()):
        truncation_rate = sum(row.finish_reason == "length" for row in rows) / len(rows)
        quality["by_dataset"][dataset] = {
            "samples": len(rows),
            "questions": len({row.question_id for row in rows}),
            "metric": rows[0].metric,
            "score": mean(row.score or 0.0 for row in rows),
            "parse_fail_rate": sum(row.parse_failed for row in rows) / len(rows),
            "truncation_rate": truncation_rate,
            "quality_valid": truncation_rate <= 0.05,
        }
    for category, rows in sorted(grouped_category.items()):
        dataset_scores = [
            quality["by_dataset"][dataset]["score"]
            for dataset in sorted({row.dataset for row in rows})
        ]
        quality["by_category"][category] = {
            "datasets": sorted({row.dataset for row in rows}),
            "samples": len(rows),
            "macro_mean_score": mean(dataset_scores),
            "sample_mean_score": mean(row.score or 0.0 for row in rows),
        }
    for question_type, rows in sorted(grouped_type.items()):
        quality["by_question_type"][question_type] = {
            "samples": len(rows),
            "score": mean(row.score or 0.0 for row in rows),
        }
    for metric, rows in sorted(grouped_metric.items()):
        quality["by_metric"][metric] = {
            "samples": len(rows),
            "score": mean(row.score or 0.0 for row in rows),
        }

    binary_questions = {
        key: rows
        for key, rows in grouped_question.items()
        if rows and rows[0].metric in BINARY_METRICS
    }
    max_samples = max((len(rows) for rows in binary_questions.values()), default=0)
    if max_samples > 1:
        first_scores = []
        pass_scores = []
        majority_scores = []
        consistency = []
        for rows in binary_questions.values():
            ordered = sorted(rows, key=lambda row: row.sample_id)
            first_scores.append(ordered[0].score or 0.0)
            correct = sum((row.score or 0.0) == 1.0 for row in ordered)
            n = len(ordered)
            k = n
            pass_scores.append(
                1.0 - (comb(n - correct, k) / comb(n, k) if n - correct >= k else 0.0)
            )
            answers = [row.parsed_answer for row in ordered if row.parsed_answer is not None]
            if answers:
                majority = Counter(answers).most_common(1)[0][0]
                matching = next((row for row in ordered if row.parsed_answer == majority), None)
                majority_scores.append(float(bool(matching and (matching.score or 0.0) > 0)))
                consistency.append(float(len(set(answers)) == 1 and len(answers) == len(ordered)))
            else:
                majority_scores.append(0.0)
                consistency.append(0.0)
        quality.update(
            {
                "accuracy_at_1": mean(first_scores),
                "pass_at_k": mean(pass_scores),
                "pass_k": max_samples,
                "majority_at_k": mean(majority_scores),
                "consistency_at_k": mean(consistency),
            }
        )
    invalid = [
        dataset for dataset, values in quality["by_dataset"].items() if not values["quality_valid"]
    ]
    quality["quality_valid"] = not invalid
    quality["invalid_datasets"] = invalid

    performance = {
        "total_requests": total,
        "supported_requests": len(supported),
        "unsupported_requests": len(unsupported),
        "successful_requests": len(successful),
        "failed_requests": len(supported) - len(successful),
        "elapsed_seconds": elapsed_seconds,
        "qps": len(supported) / elapsed_seconds if elapsed_seconds > 0 else 0.0,
        "successful_qps": len(successful) / elapsed_seconds if elapsed_seconds > 0 else 0.0,
        "input_tokens_per_second": (
            sum(row.input_tokens for row in successful) / elapsed_seconds
            if elapsed_seconds > 0
            else 0.0
        ),
        "output_tokens_per_second": (
            sum(row.output_tokens for row in successful) / elapsed_seconds
            if elapsed_seconds > 0
            else 0.0
        ),
        "error_rate": ((len(supported) - len(successful)) / len(supported) if supported else 0.0),
        "timeout_rate": errors.get("timeout", 0) / len(supported) if supported else 0.0,
        "truncated_responses": sum(row.finish_reason == "length" for row in successful),
        "truncation_rate": (
            sum(row.finish_reason == "length" for row in successful) / len(successful)
            if successful
            else 0.0
        ),
        "latency_ms": _distribution(latency),
        "final_attempt_latency_ms": _distribution(attempt_latency),
        "ttft_ms": _distribution(ttft),
        "tpot_ms": _distribution(tpot),
        "errors": dict(sorted(errors.items())),
    }
    return {
        "schema_version": 2,
        "run_id": run_id,
        "mode": mode,
        "model": model,
        "base_url": base_url,
        "config": config,
        "quality": quality,
        "performance": performance,
    }
