#!/usr/bin/env python3
"""Verify the self-contained core wheel, including every bundled PNG resource."""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import re
import zipfile
from collections import Counter
from email.parser import BytesParser
from pathlib import Path

# Existing text corpus hashes are independent of the wheel's manifest.
TEXT_DATASETS = {
    "mmlu-pro": (500, "629a9fa1e42314bdff4103851c18dcb52c81a8d32b0ccbf023186f1a72b2c955"),
    "mmlu-redux": (500, "f4391f63ac49e1685422733ff52e78c084b49f9f071da3fe7079c8a798545fd4"),
    "gpqa-diamond": (100, "acc2cd4cf2d9905991c05af10ae41bbf897b58fd046afece201cfd646b0048ef"),
    "gsm8k": (500, "73a2ec0c5f7a218c48fea30341240747b765ec1e59a74d488928867f89f5116b"),
    "ceval": (500, "3698a2b3b95b0a729f653ba30f484f2d0d4227c66a5ad61d174c89d54ab0d703"),
    "hellaswag": (500, "0e398562890bad9116c5b10c89da789a2457f46bdc76f8e75b6ca00df7c47ebc"),
    "truthfulqa": (200, "62fc370ee3dc2d3c9416380804bd5c3aa0d6ae34a7f530b5c75a7960b18b76d5"),
    "drop": (500, "bda9e07d59a481343385debc17071ae137275abc1fda7cecb1dec0592ad91703"),
}


def verify_wheel(path: Path) -> dict[str, int]:
    with zipfile.ZipFile(path) as wheel:
        names = set(wheel.namelist())
        manifest = json.loads(wheel.read("llmbench/data/manifest.json"))
        catalog = json.loads(wheel.read("llmbench/data/benchmark_catalog.json"))
        assert set(manifest) == {*TEXT_DATASETS, "stress", "mmmu"}, "bundled dataset IDs"
        assert catalog["count"] == len(catalog["benchmarks"]) == 155
        records = {}
        for name, metadata in manifest.items():
            resource_name = f"llmbench/data/{metadata['file']}"
            assert resource_name in names, resource_name
            payload = wheel.read(resource_name)
            digest = hashlib.sha256(payload).hexdigest()
            assert digest == metadata["sha256"], f"{name}: JSONL SHA-256"
            rows = [json.loads(line) for line in payload.split(b"\n") if line.strip()]
            assert len(rows) == metadata["count"], f"{name}: record count"
            assert len({row["id"] for row in rows}) == len(rows), f"{name}: unique IDs"
            if name in TEXT_DATASETS:
                assert (len(rows), digest) == TEXT_DATASETS[name], f"{name}: unchanged text corpus"
            records[name] = rows
        assert sum(len(records[name]) for name in TEXT_DATASETS) == 3300
        assert len(records["stress"]) == 10
        assert manifest["stress"]["sha256"] == (
            "c3befb9130ae6d14a3b060906f872f65578ba41578dcd55499fb485ae44b73fe"
        )

        vision = manifest["mmmu"]
        assert len(records["mmmu"]) == 500
        assert vision["capability"] == "multimodal"
        assert vision["adapter"] == "multimodal_chat"
        assert vision["metric"] == "accuracy"
        assert vision["license"] == "Apache-2.0"
        assert vision["source_revision"] == "98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68"
        assert vision["source"].endswith(vision["source_revision"])
        assert vision["selection"]["count"] == 500
        assert vision["provenance"]["source_jsonl_sha256"] == (
            "391548531838785781989852f5f980e1f733f8bb35fd62c51ec493d49a9c6b1e"
        )
        type_counts = Counter(row["type"] for row in records["mmmu"])
        assert type_counts == vision["types"] == {"multiple_choice": 476, "mmmu_open": 24}
        images: dict[str, int] = {}
        reference_count = 0
        for row in records["mmmu"]:
            assert row["type"] == ("multiple_choice" if row.get("choices") else "mmmu_open")
            metadata = row["metadata"]
            assert metadata["capability"] == "multimodal"
            assert metadata["adapter"] == "multimodal_chat"
            assert metadata["benchmark_metric"] == "accuracy"
            assert metadata["assets"], f"{row['id']}: image assets"
            assert set(metadata["asset_sha256"]) == set(metadata["assets"])
            for asset in metadata["assets"]:
                reference_count += 1
                digest = metadata["asset_sha256"][asset]
                assert re.fullmatch(r"[a-f0-9]{64}", digest), f"{row['id']}: image digest"
                assert asset == f"data/images/mmmu/{digest}.png", f"{row['id']}: asset path"
                resource_name = f"llmbench/{asset}"
                assert resource_name in names, f"missing image resource: {resource_name}"
                if resource_name not in images:
                    payload = wheel.read(resource_name)
                    assert payload.startswith(b"\x89PNG\r\n\x1a\n"), resource_name
                    assert hashlib.sha256(payload).hexdigest() == digest, resource_name
                    images[resource_name] = len(payload)
        assert reference_count == vision["image_reference_count"] == 541
        assert len(images) == vision["image_count"] == 535
        assert sum(images.values()) == vision["image_bytes"] == 188745878
        assert {name for name in names if name.startswith("llmbench/data/images/")} == set(images)
        assert not any(name.startswith("llmbench_data_") for name in names), "legacy data package"

        metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
        assert len(metadata_names) == 1, "single distribution metadata"
        package_metadata = BytesParser().parsebytes(wheel.read(metadata_names[0]))
        assert package_metadata["Name"] == "llm-bench"
        assert "image" in package_metadata.get_all("Provides-Extra", [])
        requirements = package_metadata.get_all("Requires-Dist", [])
        image_requirements = [item for item in requirements if item.lower().startswith("pillow")]
        assert image_requirements, "Pillow image extra dependency"
        assert any(re.search(r"extra\s*==\s*['\"]image['\"]", item) for item in image_requirements)
        assert all(
            re.search(r"extra\s*==\s*['\"](?:image|dev)['\"]", item) for item in image_requirements
        ), "Pillow must remain optional, never a base runtime dependency"
        assert not any("llmbench-data" in item.lower() for item in requirements)
        for name in names:
            if name.endswith(".dist-info/entry_points.txt"):
                entry_points = configparser.ConfigParser()
                entry_points.read_string(wheel.read(name).decode())
                assert not entry_points.has_section("llmbench.data_packs"), "core data entry point"
    return {"text": 3300, "stress": 10, "vision": 500, "images": len(images)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    counts = verify_wheel(args.wheel)
    print(
        f"Verified {args.wheel}: {counts['text']} text questions, "
        f"{counts['stress']} stress prompts, {counts['vision']} vision questions, "
        f"{counts['images']} unique PNGs; all hashes and optional image dependencies valid"
    )


if __name__ == "__main__":
    main()
