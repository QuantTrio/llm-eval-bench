from __future__ import annotations

import hashlib
import json
from typing import Any

from . import __version__
from .datasets import dataset_metadata
from .schemas import DatasetItem


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_run_manifest(
    *,
    run_id: str,
    mode: str,
    model: str,
    base_url: str,
    config: dict[str, Any],
    items: list[DatasetItem],
    n_samples: int,
) -> dict[str, Any]:
    question_keys = [
        [item.dataset, item.id, sample_id]
        for item in items
        for sample_id in range(1, n_samples + 1)
    ]
    datasets = {}
    for name in config.get("datasets", []):
        metadata = dataset_metadata(name)
        datasets[name] = {
            "sha256": metadata.get("sha256"),
            "count": metadata.get("count"),
            "metric": metadata.get("metric"),
            "category": metadata.get("category"),
        }
    comparable_config = {
        key: value
        for key, value in config.items()
        if key not in {"available_models", "memory_gb", "progress_interval", "checkpoint_every"}
    }
    fingerprint_input = {
        "mode": mode,
        "model": model,
        "base_url": base_url,
        "config": comparable_config,
        "question_keys": question_keys,
        "datasets": datasets,
    }
    return {
        "schema_version": 2,
        "run_id": run_id,
        "llmbench_version": __version__,
        "mode": mode,
        "model": model,
        "base_url": base_url,
        "target_capabilities": ["chat", "stream"] if config.get("stream") else ["chat"],
        "config": comparable_config,
        "datasets": datasets,
        "question_count": len(items),
        "request_count": len(question_keys),
        "question_keys_sha256": canonical_hash(question_keys),
        "fingerprint": canonical_hash(fingerprint_input),
    }
