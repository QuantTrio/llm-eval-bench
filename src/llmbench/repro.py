from __future__ import annotations

import hashlib
import json
from typing import Any

from . import __version__
from .datasets import dataset_metadata
from .images import prepare_image_messages
from .schemas import DatasetItem
from .scoring import build_messages


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
    prompt_inputs = []
    image_inputs = []
    for item in items:
        if item.is_image:
            messages, assets = prepare_image_messages(item)
            if not assets:
                raise ValueError(f"image question {item.id!r} has no image assets")
            messages_hash = canonical_hash(messages)
            prompt_inputs.append([item.dataset, item.id, {"images_sha256": messages_hash}])
            image_inputs.append(
                {
                    "dataset": item.dataset,
                    "question_id": item.id,
                    "messages_sha256": messages_hash,
                    "assets": assets,
                }
            )
            # Retain hashes only, not all images' base64 strings, during a large run.
            del messages
        else:
            prompt_inputs.append([item.dataset, item.id, build_messages(item)])
    fingerprint_input = {
        "mode": mode,
        "model": model,
        "base_url": base_url,
        "config": comparable_config,
        "question_keys": question_keys,
        "datasets": datasets,
        "prompts_sha256": canonical_hash(prompt_inputs),
    }
    manifest = {
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
        "prompts_sha256": fingerprint_input["prompts_sha256"],
        "fingerprint": canonical_hash(fingerprint_input),
    }
    if image_inputs:
        manifest["image_inputs"] = image_inputs
        manifest["target_capabilities"].append("image")
    return manifest
