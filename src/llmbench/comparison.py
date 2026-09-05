from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .repro import canonical_hash
from .schemas import RequestResult


class IncomparableRunsError(ValueError):
    pass


def _run_directory(path: Path) -> Path:
    return path if path.is_dir() else path.parent


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_results(path: Path) -> list[RequestResult]:
    results = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                results.append(RequestResult.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"Invalid result at {path}:{line_number}: {exc}") from exc
    return results


def load_run(path: Path) -> dict[str, Any]:
    directory = _run_directory(path)
    summary_path = (
        path if path.is_file() and path.name == "summary.json" else directory / "summary.json"
    )
    manifest_path = directory / "run_manifest.json"
    raw_path = directory / "raw_results.jsonl"
    missing = [
        candidate for candidate in (summary_path, manifest_path, raw_path) if not candidate.exists()
    ]
    if missing:
        raise ValueError(
            "Run is missing required artifacts: " + ", ".join(str(item) for item in missing)
        )
    results = _load_results(raw_path)
    keys = [result.key for result in results]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Run contains duplicate request keys: {directory}")
    return {
        "directory": directory,
        "summary": _load_json(summary_path),
        "manifest": _load_json(manifest_path),
        "results": results,
    }


def _comparison_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    config = {
        key: value
        for key, value in manifest.get("config", {}).items()
        if key
        not in {
            "available_models",
            "memory_gb",
            "progress_interval",
            "checkpoint_every",
            "provider",
            "api_key_env",
        }
    }
    return {
        "mode": manifest.get("mode"),
        "config": config,
        "datasets": manifest.get("datasets"),
        "question_keys_sha256": manifest.get("question_keys_sha256"),
        "prompts_sha256": manifest.get("prompts_sha256"),
        "request_count": manifest.get("request_count"),
    }


def _bootstrap_interval(
    deltas: list[float], *, samples: int = 10_000, seed: int = 42
) -> dict[str, float | int | None]:
    if not deltas:
        return {"lower": None, "upper": None, "confidence": 0.95, "samples": samples}
    rng = random.Random(seed)
    values = sorted(mean(rng.choice(deltas) for _ in deltas) for _ in range(samples))
    lower = values[int(samples * 0.025)]
    upper = values[min(int(samples * 0.975), samples - 1)]
    return {"lower": lower, "upper": upper, "confidence": 0.95, "samples": samples}


def _relative_change(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline in {None, 0}:
        return None
    return (candidate - baseline) / baseline


def compare_run_directories(
    baseline_path: Path,
    candidate_path: Path,
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    baseline = load_run(baseline_path)
    candidate = load_run(candidate_path)
    for run in (baseline, candidate):
        expected = run["manifest"].get("request_count")
        if expected is not None and len(run["results"]) != expected:
            raise IncomparableRunsError(f"run is incomplete: {run['directory']}")
    base_payload = _comparison_payload(baseline["manifest"])
    cand_payload = _comparison_payload(candidate["manifest"])
    if canonical_hash(base_payload) != canonical_hash(cand_payload):
        differences = [
            key
            for key in sorted(set(base_payload) | set(cand_payload))
            if base_payload.get(key) != cand_payload.get(key)
        ]
        raise IncomparableRunsError(
            "run manifests differ in comparison-critical fields: " + ", ".join(differences)
        )

    base_by_key = {result.key: result for result in baseline["results"]}
    cand_by_key = {result.key: result for result in candidate["results"]}
    if set(base_by_key) != set(cand_by_key):
        missing_candidate = len(set(base_by_key) - set(cand_by_key))
        missing_baseline = len(set(cand_by_key) - set(base_by_key))
        raise IncomparableRunsError(
            f"paired request keys differ: missing_candidate={missing_candidate}, "
            f"missing_baseline={missing_baseline}"
        )

    paired: list[dict[str, Any]] = []
    by_dataset: dict[str, list[float]] = defaultdict(list)
    transitions: Counter[str] = Counter()
    for key in sorted(base_by_key):
        base = base_by_key[key]
        cand = cand_by_key[key]
        if base.prompt != cand.prompt:
            raise IncomparableRunsError(f"prompt differs for request {key}")
        if base.metric != cand.metric or base.gold_answer != cand.gold_answer:
            raise IncomparableRunsError(f"metric or gold answer differs for request {key}")
        if base.score is None or cand.score is None:
            delta = None
            transition = "unscored"
        else:
            delta = cand.score - base.score
            by_dataset[base.dataset].append(delta)
            if base.score == 1 and cand.score == 0:
                transition = "correct_to_wrong"
            elif base.score == 0 and cand.score == 1:
                transition = "wrong_to_correct"
            elif delta == 0:
                transition = "unchanged"
            else:
                transition = "partial_score_changed"
        transitions[transition] += 1
        paired.append(
            {
                "dataset": base.dataset,
                "question_id": base.question_id,
                "sample_id": base.sample_id,
                "metric": base.metric,
                "baseline_score": base.score,
                "candidate_score": cand.score,
                "delta": delta,
                "transition": transition,
                "baseline_answer": base.parsed_answer,
                "candidate_answer": cand.parsed_answer,
                "baseline_raw_output": base.raw_output,
                "candidate_raw_output": cand.raw_output,
                "gold_answer": base.gold_answer,
            }
        )

    dataset_comparison = {}
    for dataset, deltas in sorted(by_dataset.items()):
        base_values = [
            row for row in baseline["results"] if row.dataset == dataset and row.score is not None
        ]
        cand_values = [
            row for row in candidate["results"] if row.dataset == dataset and row.score is not None
        ]
        base_score = mean(row.score or 0.0 for row in base_values)
        cand_score = mean(row.score or 0.0 for row in cand_values)
        dataset_comparison[dataset] = {
            "metric": base_values[0].metric,
            "baseline": base_score,
            "candidate": cand_score,
            "absolute_change": mean(deltas),
            "relative_change": _relative_change(cand_score, base_score),
            "confidence_interval": _bootstrap_interval(
                deltas, samples=bootstrap_samples, seed=seed
            ),
        }

    all_deltas = [row["delta"] for row in paired if row["delta"] is not None]
    base_summary = baseline["summary"]
    cand_summary = candidate["summary"]
    base_perf = base_summary["performance"]
    cand_perf = cand_summary["performance"]
    base_p95 = base_perf["latency_ms"].get("p95")
    cand_p95 = cand_perf["latency_ms"].get("p95")
    return {
        "schema_version": 2,
        "comparable": True,
        "comparison_fingerprint": canonical_hash(base_payload),
        "baseline": {
            "run_id": base_summary["run_id"],
            "model": base_summary["model"],
            "directory": str(baseline["directory"]),
        },
        "candidate": {
            "run_id": cand_summary["run_id"],
            "model": cand_summary["model"],
            "directory": str(candidate["directory"]),
        },
        "quality": {
            "sample_mean_change": mean(all_deltas) if all_deltas else None,
            "confidence_interval": _bootstrap_interval(
                all_deltas, samples=bootstrap_samples, seed=seed
            ),
            "transitions": dict(sorted(transitions.items())),
            "by_dataset": dataset_comparison,
        },
        "performance": {
            "throughput_change": _relative_change(
                cand_perf.get("output_tokens_per_second"),
                base_perf.get("output_tokens_per_second"),
            ),
            "qps_change": _relative_change(cand_perf.get("qps"), base_perf.get("qps")),
            "p95_latency_change": _relative_change(cand_p95, base_p95),
            "error_rate": cand_perf.get("error_rate", 0),
            "error_rate_change": cand_perf.get("error_rate", 0) - base_perf.get("error_rate", 0),
            "truncation_rate": cand_perf.get("truncation_rate", 0),
            "truncation_rate_change": cand_perf.get("truncation_rate", 0)
            - base_perf.get("truncation_rate", 0),
        },
        "paired_results": paired,
    }


def evaluate_policy(comparison: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    gates = policy.get("gates") or {}
    results: list[dict[str, Any]] = []

    def maximum(name: str, actual: float | None, limit: float | None) -> None:
        if limit is None:
            return
        passed = actual is not None and actual <= limit
        results.append({"gate": name, "actual": actual, "limit": limit, "passed": passed})

    def minimum(name: str, actual: float | None, limit: float | None) -> None:
        if limit is None:
            return
        passed = actual is not None and actual >= limit
        results.append({"gate": name, "actual": actual, "limit": limit, "passed": passed})

    quality = comparison["quality"]
    performance = comparison["performance"]
    overall_change = quality.get("sample_mean_change")
    maximum(
        "max_score_drop",
        None if overall_change is None else -overall_change,
        gates.get("max_score_drop"),
    )
    for dataset, values in quality.get("by_dataset", {}).items():
        maximum(
            f"max_dataset_drop:{dataset}",
            -values["absolute_change"],
            gates.get("max_dataset_drop"),
        )
    maximum("max_error_rate", performance.get("error_rate"), gates.get("max_error_rate"))
    maximum(
        "max_truncation_rate",
        performance.get("truncation_rate"),
        gates.get("max_truncation_rate"),
    )
    maximum(
        "max_p95_latency_increase",
        performance.get("p95_latency_change"),
        gates.get("max_p95_latency_increase"),
    )
    minimum(
        "min_throughput_change",
        performance.get("throughput_change"),
        gates.get("min_throughput_change"),
    )
    return {"passed": all(item["passed"] for item in results), "gates": results}
