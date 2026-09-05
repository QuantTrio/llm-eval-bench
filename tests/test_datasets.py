from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from pathlib import Path

import pytest

from llmbench import data_packs
from llmbench.catalog import benchmark_catalog, installed_datasets, list_benchmarks
from llmbench.datasets import dataset_metadata, load_dataset, load_many, stress_prompts


@pytest.fixture(autouse=True)
def no_optional_data_packs(monkeypatch) -> None:
    """Core dataset tests must be independent of developer-installed old packs."""
    monkeypatch.setattr(data_packs, "_entry_points", lambda: [])


def test_bundled_dataset_manifest_and_checksums(monkeypatch) -> None:
    monkeypatch.setattr("llmbench.data_packs._entry_points", lambda: [])
    catalog = {item["name"]: item for item in installed_datasets()}
    assert set(catalog) == {
        "ceval",
        "drop",
        "gpqa-diamond",
        "gsm8k",
        "hellaswag",
        "mmlu-pro",
        "mmlu-redux",
        "mmmu",
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
    assert item.metadata["asset_base_dir"] == str(tmp_path.resolve())
    with pytest.raises(ValueError, match="Unknown dataset"):
        load_dataset("does-not-exist")


def test_bundled_mmmu_resources_and_question_types() -> None:
    items = load_dataset("mmmu")
    metadata = dataset_metadata("mmmu")
    assert len(items) == 500
    assert sum(bool(item.choices) for item in items) == 476
    assert sum(item.type == "mmmu_open" for item in items) == 24
    assert metadata["capability"] == "multimodal"
    assert metadata["metric"] == "accuracy"
    image_hashes = {}
    image_references = 0
    for item in items:
        assert item.type == ("multiple_choice" if item.choices else "mmmu_open")
        assert item.metadata["resource_package"] == "llmbench"
        assert item.metadata["capability"] == "multimodal"
        assert item.metadata["adapter"] == "multimodal_chat"
        assert item.metadata["benchmark_metric"] == "accuracy"
        for asset in item.metadata["assets"]:
            image_references += 1
            expected = item.metadata["asset_sha256"][asset]
            assert asset == f"data/images/mmmu/{expected}.png"
            if asset not in image_hashes:
                payload = files("llmbench").joinpath(asset).read_bytes()
                assert payload.startswith(b"\x89PNG\r\n\x1a\n")
                image_hashes[asset] = hashlib.sha256(payload).hexdigest()
            assert image_hashes[asset] == expected
    assert len(image_hashes) == metadata["image_count"] == 535
    assert image_references == metadata["image_reference_count"] == 541


def test_core_datasets_do_not_discover_optional_packs(monkeypatch) -> None:
    def unavailable():
        raise AssertionError("Loading built-in datasets must not discover optional packs")

    monkeypatch.setattr(data_packs, "_entry_points", unavailable)
    assert dataset_metadata("mmmu")["count"] == 500
    assert load_dataset("mmmu", limit=1)[0].metadata["resource_package"] == "llmbench"


def test_core_ids_take_precedence_over_multiple_legacy_packs(monkeypatch) -> None:
    packs = [
        {
            "name": f"legacy-{index}",
            "version": "0.5.0",
            "package": "old_package_that_must_not_be_loaded",
            "datasets": {"mmmu": {"file": "old.jsonl", "count": 999}},
        }
        for index in range(2)
    ]
    monkeypatch.setattr(data_packs, "discover_data_packs", lambda: packs)
    metadata = {item["name"]: item for item in installed_datasets()}["mmmu"]
    assert metadata["count"] == 500
    assert "pack" not in metadata
    assert {"name": "mmmu", **dataset_metadata("mmmu")} == metadata
    assert load_dataset("mmmu", limit=1)[0].metadata["resource_package"] == "llmbench"
    with pytest.raises(ValueError, match="multiple installed data packs"):
        data_packs.external_dataset_resources()
    for pack in packs:
        pack["datasets"]["external-only"] = {"file": "external.jsonl"}
    with pytest.raises(ValueError, match=r"external-only.*multiple installed data packs"):
        installed_datasets()


def test_manifest_supplies_builtin_adapter_defaults(monkeypatch, tmp_path) -> None:
    package_dir = tmp_path / "package"
    data_dir = package_dir / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "fixture.jsonl").write_text(
        json.dumps({"id": "fixture", "type": "mmmu_open", "question": "Read the picture"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("llmbench.datasets.files", lambda _package: package_dir)
    monkeypatch.setattr(
        "llmbench.datasets.read_manifest",
        lambda: {
            "fixture": {
                "file": "fixture.jsonl",
                "category": "Vision",
                "metric": "accuracy",
                "capability": "multimodal",
                "adapter": "multimodal_chat",
            }
        },
    )
    item = load_dataset("fixture")[0]
    assert item.metadata["resource_package"] == "llmbench"
    assert item.metadata["capability"] == "multimodal"
    assert item.metadata["adapter"] == "multimodal_chat"


def test_local_asset_base_is_independent_of_current_directory(monkeypatch, tmp_path) -> None:
    dataset_dir = tmp_path / "fixture"
    dataset_dir.mkdir()
    source = dataset_dir / "vision.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "local-image",
                "type": "exact_match",
                "question": "What is shown?",
                "metadata": {"assets": ["image.png"]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    item = load_dataset("fixture/vision.jsonl")[0]
    assert Path(item.metadata["asset_base_dir"]) == dataset_dir.resolve()


def test_datalearner_catalog_snapshot() -> None:
    snapshot = benchmark_catalog()
    assert snapshot["count"] == 155
    top = list_benchmarks(category="科学", bundled_only=True)
    assert top[0]["code"] == "gpqa-diamond"
    assert top[0]["report_count"] >= 200


def test_stress_prompt_profiles() -> None:
    assert stress_prompts("short")[0].metadata["prompt_profile"] == "short"
    assert len(stress_prompts("medium")[0].question) > len(stress_prompts("short")[0].question)
    assert stress_prompts("mixed")[0].dataset == "stress"
    with pytest.raises(ValueError, match="prompt profile"):
        stress_prompts("invalid")
