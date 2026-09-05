from __future__ import annotations

import hashlib
import json
import random
from importlib.resources import files
from pathlib import Path
from typing import Any

from .data_packs import external_dataset_resources
from .schemas import DatasetItem


def read_manifest() -> dict[str, dict[str, Any]]:
    """The metadata for datasets bundled in the wheel."""
    resource = files("llmbench").joinpath("data", "manifest.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def dataset_metadata(name_or_path: str) -> dict[str, Any]:
    candidate = Path(name_or_path).expanduser()
    if candidate.exists():
        payload = candidate.read_bytes()
        return {
            "count": sum(bool(line.strip()) for line in payload.splitlines()),
            "category": "Custom",
            "metric": "custom",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "source": str(candidate.resolve()),
            "recommended_max_tokens": 4096,
        }
    manifest = read_manifest()
    metadata = manifest.get(name_or_path)
    if metadata is None:
        external = external_dataset_resources(exclude_names=manifest).get(name_or_path)
        metadata = external.metadata if external else None
    if metadata is None:
        raise ValueError(f"Unknown dataset '{name_or_path}'")
    return dict(metadata)


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
            item.metadata["asset_base_dir"] = str(candidate.parent.resolve())
            item.metadata.setdefault("benchmark_category", "Custom")
            item.metadata.setdefault(
                "benchmark_metric",
                "token_f1" if item.type == "f1" else "exact_match",
            )
            item.metadata.setdefault("recommended_max_tokens", 4096)
    else:
        manifest = read_manifest()
        if name_or_path in manifest:
            metadata = manifest[name_or_path]
            resource_package = "llmbench"
            resource = files(resource_package).joinpath("data", metadata["file"])
        else:
            external = external_dataset_resources(exclude_names=manifest)
            if name_or_path not in external:
                available = ", ".join(sorted(set(manifest) | set(external)))
                raise ValueError(
                    f"Unknown dataset '{name_or_path}'. Built-ins: {available}; "
                    "or pass a local JSONL path."
                )
            external_resource = external[name_or_path]
            metadata = external_resource.metadata
            resource_package = external_resource.package
            resource = files(resource_package).joinpath(external_resource.file)
        items = _read_jsonl(resource, source=name_or_path)
        for item in items:
            item.metadata["resource_package"] = resource_package
            for key in ("capability", "adapter"):
                if key in metadata:
                    item.metadata.setdefault(key, metadata[key])
            item.metadata.setdefault("benchmark_category", metadata["category"])
            item.metadata.setdefault("benchmark_metric", metadata["metric"])
            item.metadata.setdefault(
                "recommended_max_tokens", metadata.get("recommended_max_tokens", 4096)
            )

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


def stress_prompts(profile: str) -> list[DatasetItem]:
    if profile == "mixed":
        return load_dataset("stress")
    lengths = {"short": 32, "medium": 512, "long": 4096}
    if profile not in lengths:
        raise ValueError("prompt profile must be one of: short, medium, long, mixed")
    words = [
        "Evaluate",
        "serving",
        "performance",
        "with",
        "deterministic",
        "synthetic",
        "context.",
    ]
    target = lengths[profile]
    content = " ".join(words[index % len(words)] for index in range(target))
    return [
        DatasetItem(
            id=f"stress-{profile}",
            dataset="stress",
            type="stress",
            question=f"{content}\nSummarize the request in one sentence.",
            metadata={
                "benchmark_category": "Performance",
                "benchmark_metric": "none",
                "prompt_profile": profile,
                "recommended_max_tokens": 128,
            },
        )
    ]
