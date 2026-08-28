from __future__ import annotations

import json
import random
from importlib.resources import files
from pathlib import Path
from typing import Any

from .schemas import DatasetItem


def _manifest() -> dict[str, dict[str, Any]]:
    resource = files("llmbench").joinpath("data", "manifest.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def list_datasets() -> list[dict[str, Any]]:
    return [dict(name=name, **metadata) for name, metadata in sorted(_manifest().items())]


def _read_jsonl(path: Any, *, source: str) -> list[DatasetItem]:
    items: list[DatasetItem] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                items.append(DatasetItem.from_dict(value, source=source))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return items


def load_dataset(
    name_or_path: str,
    *,
    limit: int | None = None,
    sample: int | None = None,
    seed: int = 42,
) -> list[DatasetItem]:
    candidate = Path(name_or_path).expanduser()
    if candidate.exists():
        items = _read_jsonl(candidate, source=candidate.stem)
        for item in items:
            item.metadata.setdefault("benchmark_category", "Custom")
            item.metadata.setdefault(
                "benchmark_metric",
                "token_f1" if item.type == "f1" else "exact_match",
            )
    else:
        manifest = _manifest()
        if name_or_path not in manifest:
            available = ", ".join(sorted(manifest))
            raise ValueError(
                f"Unknown dataset '{name_or_path}'. Built-ins: {available}; "
                "or pass a local JSONL path."
            )
        metadata = manifest[name_or_path]
        resource = files("llmbench").joinpath("data", metadata["file"])
        items = _read_jsonl(resource, source=name_or_path)
        for item in items:
            item.metadata.setdefault("benchmark_category", metadata["category"])
            item.metadata.setdefault("benchmark_metric", metadata["metric"])

    if sample is not None:
        if sample < 1:
            raise ValueError("sample must be at least 1")
        rng = random.Random(seed)
        items = rng.sample(items, min(sample, len(items)))
    elif limit is not None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        items = items[:limit]
    return items


def load_many(
    dataset_specs: list[str],
    *,
    limit_per_dataset: int | None,
    sample: int | None,
    seed: int,
) -> list[DatasetItem]:
    items: list[DatasetItem] = []
    for offset, spec in enumerate(dataset_specs):
        items.extend(load_dataset(spec, limit=limit_per_dataset, sample=sample, seed=seed + offset))
    return items
