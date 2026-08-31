from __future__ import annotations

import hashlib
import json
from importlib.resources import files

import pytest

from llmbench.catalog import benchmark_catalog, list_benchmarks
from llmbench.datasets import list_datasets, load_dataset, load_many, stress_prompts


def test_bundled_dataset_manifest_and_checksums(monkeypatch) -> None:
    monkeypatch.setattr("llmbench.data_packs._entry_points", lambda: [])
    catalog = {item["name"]: item for item in list_datasets()}
    assert set(catalog) == {
        "ceval",
        "drop",
        "gpqa-diamond",
        "gsm8k",
        "hellaswag",
        "mmlu-pro",
        "mmlu-redux",
        "stress",
        "truthfulqa",
    }
    for name, metadata in catalog.items():
        resource = files("llmbench").joinpath("data", metadata["file"])
        digest = hashlib.sha256(resource.read_bytes()).hexdigest()
        assert digest == metadata["sha256"], name
        assert len(load_dataset(name)) == metadata["count"]


def test_limit_sample_and_category_metadata_are_deterministic() -> None:
    first = load_dataset("mmlu-pro", sample=5, seed=7)
    second = load_dataset("mmlu-pro", sample=5, seed=7)
    assert [item.id for item in first] == [item.id for item in second]
    assert all(item.metadata["benchmark_category"] == "Comprehensive" for item in first)
    assert len(load_dataset("mmlu-pro", limit=3, sample=5, seed=7)) == 5
    assert len(load_many(["gsm8k", "gpqa-diamond"], limit_per_dataset=3, sample=None, seed=42)) == 6


def test_custom_jsonl_and_invalid_dataset(tmp_path) -> None:
    path = tmp_path / "custom.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "custom-1",
                "type": "exact_match",
                "question": "Capital of France?",
                "answer": "Paris",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    item = load_dataset(str(path))[0]
    assert item.dataset == "custom"
    assert item.metadata["benchmark_category"] == "Custom"
    with pytest.raises(ValueError, match="Unknown dataset"):
        load_dataset("does-not-exist")


def test_datalearner_catalog_snapshot() -> None:
    snapshot = benchmark_catalog()
    assert snapshot["count"] == 157
    top = list_benchmarks(category="科学", bundled_only=True)
    assert top[0]["code"] == "gpqa-diamond"
    assert top[0]["report_count"] >= 200


def test_stress_prompt_profiles() -> None:
    assert stress_prompts("short")[0].metadata["prompt_profile"] == "short"
    assert len(stress_prompts("medium")[0].question) > len(stress_prompts("short")[0].question)
    assert stress_prompts("mixed")[0].dataset == "stress"
    with pytest.raises(ValueError, match="prompt profile"):
        stress_prompts("invalid")
