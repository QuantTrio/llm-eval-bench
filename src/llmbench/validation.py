from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any

import jsonschema


def schema(name: str) -> dict[str, Any]:
    return json.loads(files("llmbench").joinpath("schema", name).read_text(encoding="utf-8"))


def validate_run_directory(directory: Path) -> list[str]:
    checks = [
        (directory / "summary.json", "summary-v2.json"),
        (directory / "run_manifest.json", "run-manifest-v2.json"),
    ]
    validated = []
    for path, schema_name in checks:
        if not path.exists():
            raise ValueError(f"missing required run artifact: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        try:
            jsonschema.Draft202012Validator(schema(schema_name)).validate(payload)
        except jsonschema.ValidationError as exc:
            location = ".".join(str(value) for value in exc.absolute_path) or "root"
            raise ValueError(f"{path.name} schema error at {location}: {exc.message}") from exc
        validated.append(path.name)
    raw = directory / "raw_results.jsonl"
    if not raw.exists():
        raise ValueError(f"missing required run artifact: {raw}")
    for line_number, line in enumerate(raw.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid raw result at line {line_number}: {exc}") from exc
        for key in ("dataset", "question_id", "sample_id", "run_id"):
            if key not in value:
                raise ValueError(f"raw result line {line_number} is missing {key}")
    validated.append(raw.name)
    return validated
