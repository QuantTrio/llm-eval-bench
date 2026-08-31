from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from llmbench.cli import app
from llmbench.validation import schema, validate_run_directory


def valid_summary() -> dict:
    return {
        "schema_version": 2,
        "run_id": "run",
        "mode": "run",
        "model": "model",
        "base_url": "http://localhost/v1",
        "config": {},
        "quality": {
            "scored_samples": 0,
            "by_dataset": {},
            "by_category": {},
            "by_question_type": {},
            "quality_valid": True,
        },
        "performance": {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "error_rate": 0,
            "latency_ms": {},
        },
    }


def valid_manifest() -> dict:
    return {
        "schema_version": 2,
        "run_id": "run",
        "llmbench_version": "0.6.0",
        "mode": "run",
        "model": "model",
        "base_url": "http://localhost/v1",
        "config": {},
        "fingerprint": "a" * 64,
    }


def write(directory, name: str, payload) -> None:
    path = directory / name
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")


def test_run_schema_validation_failures(tmp_path) -> None:
    assert schema("summary-v2.json")["properties"]["schema_version"]["const"] == 2
    with pytest.raises(ValueError, match="missing required"):
        validate_run_directory(tmp_path)
    cli_result = CliRunner().invoke(app, ["validate", str(tmp_path)])
    assert cli_result.exit_code == 4

    write(tmp_path, "summary.json", {**valid_summary(), "schema_version": 1})
    write(tmp_path, "run_manifest.json", valid_manifest())
    write(tmp_path, "raw_results.jsonl", "")
    with pytest.raises(ValueError, match="schema error"):
        validate_run_directory(tmp_path)

    write(tmp_path, "summary.json", valid_summary())
    write(tmp_path, "raw_results.jsonl", "not-json\n")
    with pytest.raises(ValueError, match="invalid raw result"):
        validate_run_directory(tmp_path)

    write(tmp_path, "raw_results.jsonl", json.dumps({"dataset": "x"}) + "\n")
    with pytest.raises(ValueError, match="missing question_id"):
        validate_run_directory(tmp_path)
