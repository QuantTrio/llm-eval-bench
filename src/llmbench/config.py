from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML at {path}: {exc}") from exc
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be an object: {path}")
    return payload


def load_bench_config(path: Path) -> dict[str, Any]:
    payload = load_yaml(path)
    if payload.get("schema_version") != 2:
        raise ValueError("bench config schema_version must be 2")
    targets = payload.get("targets") or {}
    if not isinstance(targets, dict):
        raise ValueError("bench config targets must be an object")
    run = payload.get("run") or {}
    if not isinstance(run, dict):
        raise ValueError("bench config run must be an object")
    return payload


def secret_from_env(name: str | None) -> str | None:
    if not name:
        return None
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"Required secret environment variable is not set: {name}")
    return value
