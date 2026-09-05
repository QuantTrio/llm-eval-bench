#!/usr/bin/env python3
"""Bundle the existing local MMMU subset without downloading or re-encoding images.

Run from any directory with ``python scripts/bundle_image_data.py``. The input is
the maintainer's existing inline-image JSONL; normal installs need only the wheel.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import filecmp
import hashlib
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _json_bytes(value: Any, *, indent: int | None = None) -> bytes:
    return (
        json.dumps(
            value, ensure_ascii=False, indent=indent, separators=None if indent else (",", ":")
        )
        + "\n"
    ).encode("utf-8")


def _decode_png(asset: str, *, record_id: str) -> bytes:
    prefix = "data:image/png;base64,"
    if not isinstance(asset, str) or not asset.startswith(prefix):
        raise ValueError(f"{record_id}: expected an inline PNG data URL")
    try:
        payload = base64.b64decode(asset[len(prefix) :], validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"{record_id}: invalid PNG base64") from exc
    if not payload.startswith(PNG_SIGNATURE):
        raise ValueError(f"{record_id}: image bytes do not have a PNG signature")
    return payload


def bundle_image_data(
    input_jsonl: Path,
    output_dir: Path,
    *,
    source_manifest: Path | None = None,
) -> dict[str, Any]:
    """Preserve all input rows and exact PNG bytes; reject conflicting old outputs.

    ``output_dir`` is the core package's ``data`` directory. Existing manifest
    entries and text files are retained. All conversion and source validation
    finish in a temporary directory before any package outputs are modified.
    """
    input_jsonl = input_jsonl.resolve()
    output_dir = output_dir.resolve()
    source_manifest = source_manifest or input_jsonl.with_name("pack.json")
    pack = json.loads(source_manifest.read_text(encoding="utf-8"))
    original_metadata = pack["datasets"]["mmmu"]
    source_digest = hashlib.sha256()
    output_digest = hashlib.sha256()
    type_counts: Counter[str] = Counter()
    asset_sizes: dict[str, int] = {}
    image_references = 0
    record_ids: set[str] = set()

    with tempfile.TemporaryDirectory(prefix="llmbench-mmmu-") as staging:
        staged = Path(staging)
        staged_images = staged / "images" / "mmmu"
        staged_images.mkdir(parents=True)
        staged_jsonl = staged / "mmmu.jsonl"
        with input_jsonl.open("rb") as source, staged_jsonl.open("wb") as destination:
            for line_number, line in enumerate(source, start=1):
                source_digest.update(line)
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    record_id = str(row["id"])
                    if record_id in record_ids:
                        raise ValueError(f"duplicate record ID {record_id!r}")
                    record_ids.add(record_id)
                    metadata = dict(row.get("metadata") or {})
                    assets = metadata.get("assets")
                    if not isinstance(assets, list) or not assets:
                        raise ValueError(f"{record_id}: requires at least one PNG image")
                    paths = []
                    asset_sha256 = {}
                    for asset in assets:
                        payload = _decode_png(asset, record_id=record_id)
                        digest = hashlib.sha256(payload).hexdigest()
                        path = f"data/images/mmmu/{digest}.png"
                        paths.append(path)
                        asset_sha256[path] = digest
                        image_references += 1
                        if digest not in asset_sizes:
                            (staged_images / f"{digest}.png").write_bytes(payload)
                            asset_sizes[digest] = len(payload)
                    row["type"] = "multiple_choice" if row.get("choices") else "mmmu_open"
                    type_counts[row["type"]] += 1
                    row["metadata"] = {
                        **metadata,
                        "capability": "multimodal",
                        "adapter": "multimodal_chat",
                        "benchmark_metric": "accuracy",
                        "assets": paths,
                        "asset_sha256": asset_sha256,
                    }
                    converted = _json_bytes(row)
                    destination.write(converted)
                    output_digest.update(converted)
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid MMMU source at line {line_number}: {exc}") from exc

        count = len(record_ids)
        if count != original_metadata["count"]:
            raise ValueError(
                f"Source count mismatch: expected {original_metadata['count']}, got {count}"
            )
        source_sha256 = source_digest.hexdigest()
        if source_sha256 != original_metadata["sha256"]:
            raise ValueError("Source JSONL SHA-256 does not match its pack manifest")
        metadata = {
            **original_metadata,
            "file": "mmmu.jsonl",
            "count": count,
            "type": "mixed",
            "types": dict(sorted(type_counts.items())),
            "capability": "multimodal",
            "adapter": "multimodal_chat",
            "metric": "accuracy",
            "sha256": output_digest.hexdigest(),
            "source_revision": pack["source_revision"],
            "image_count": len(asset_sizes),
            "image_reference_count": image_references,
            "image_bytes": sum(asset_sizes.values()),
            "image_format": "png",
            "image_storage": "sha256-deduplicated package resources",
            "selection": {
                "method": "preserve existing subject-balanced regression subset",
                "split": "validation",
                "count": count,
                "ordering": "source JSONL order",
            },
            "provenance": {
                "source_pack": pack["name"],
                "source_pack_version": pack["version"],
                "source_jsonl_sha256": source_sha256,
                "conversion": "base64 decode only; original PNG bytes preserved",
                "builder": "scripts/bundle_image_data.py",
            },
        }
        manifest_path = output_dir / "manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        )
        manifest["mmmu"] = metadata
        generated_files = [staged_jsonl, *sorted(staged_images.iterdir())]
        # Check every existing target first, so reruns cannot partially overwrite
        # a previously generated dataset with different content.
        for generated in generated_files:
            target = output_dir / generated.relative_to(staged)
            if target.exists() and not filecmp.cmp(generated, target, shallow=False):
                raise ValueError(f"Refusing to overwrite different existing output: {target}")
        for generated in generated_files:
            target = output_dir / generated.relative_to(staged)
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(generated, target)
        manifest_bytes = _json_bytes(manifest, indent=2)
        if not manifest_path.exists() or manifest_path.read_bytes() != manifest_bytes:
            manifest_path.write_bytes(manifest_bytes)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data-packs" / "mmmu" / "llmbench_data_mmmu" / "mmmu.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "src" / "llmbench" / "data")
    parser.add_argument("--source-manifest", type=Path)
    args = parser.parse_args()
    metadata = bundle_image_data(args.input, args.output_dir, source_manifest=args.source_manifest)
    print(
        f"Bundled MMMU: {metadata['count']} records, "
        f"{metadata['image_reference_count']} image references, "
        f"{metadata['image_count']} unique PNGs, {metadata['image_bytes']} image bytes; "
        f"JSONL SHA-256 {metadata['sha256']}"
    )


if __name__ == "__main__":
    main()
