from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

VERSION = "0.5.0"
ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path(__file__).parent / "llmbench_data_all"
PUBLIC_PACKS = (
    ("agieval", "llmbench_data_agieval"),
    ("aider-polyglot", "llmbench_data_aider_polyglot"),
    ("aime-2025", "llmbench_data_aime_2025"),
    ("browsecomp", "llmbench_data_browsecomp"),
    ("humaneval", "llmbench_data_humaneval"),
    ("ifbench", "llmbench_data_ifbench"),
    ("longbench-v2", "llmbench_data_longbench_v2"),
    ("mmmu", "llmbench_data_mmmu"),
    ("pinchbench", "llmbench_data_pinchbench"),
    ("simple-bench", "llmbench_data_simple_bench"),
    ("simpleqa", "llmbench_data_simpleqa"),
    ("terminal-bench-2", "llmbench_data_terminal_bench_2"),
)


def _validate_asset(path: Path, metadata: dict) -> bytes:
    if not path.exists():
        raise FileNotFoundError(
            f"missing generated asset {path}; run the source pack's build_pack.py first"
        )
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != metadata["sha256"]:
        raise ValueError(f"SHA256 mismatch for {path}: {digest} != {metadata['sha256']}")
    count = sum(bool(line.strip()) for line in payload.splitlines())
    if count != metadata["count"]:
        raise ValueError(f"record count mismatch for {path}: {count} != {metadata['count']}")
    return payload


def _copy_legal_files(source: Path, slug: str) -> None:
    destination = PACKAGE / "licenses" / slug
    destination.mkdir(parents=True, exist_ok=True)
    legal_files = sorted([*source.glob("NOTICE*"), *source.glob("LICENSE*")])
    if not legal_files:
        raise FileNotFoundError(f"no license or notice file found in {source}")
    for path in legal_files:
        shutil.copy2(path, destination / path.name)


def main() -> None:
    PACKAGE.mkdir(parents=True, exist_ok=True)
    datasets = {}
    source_revisions = {}
    source_packs = {}
    total_bytes = 0
    for slug, package_name in PUBLIC_PACKS:
        source = ROOT / slug / package_name
        manifest_path = source / "pack.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing source manifest: {manifest_path}")
        source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_revisions[slug] = source_manifest["source_revision"]
        source_packs[slug] = {
            "name": source_manifest["name"],
            "version": source_manifest["version"],
        }
        for dataset_name, metadata in source_manifest["datasets"].items():
            if dataset_name in datasets:
                raise ValueError(f"duplicate dataset ID in aggregate pack: {dataset_name}")
            payload = _validate_asset(source / metadata["file"], metadata)
            target_name = f"{dataset_name}.jsonl"
            (PACKAGE / target_name).write_bytes(payload)
            total_bytes += len(payload)
            datasets[dataset_name] = {
                **metadata,
                "file": target_name,
                "source_pack": source_manifest["name"],
                "source_pack_version": source_manifest["version"],
            }
        _copy_legal_files(source, slug)

    expected_files = {item["file"] for item in datasets.values()}
    for path in PACKAGE.glob("*.jsonl"):
        if path.name not in expected_files:
            path.unlink()
    expected_licenses = {slug for slug, _package_name in PUBLIC_PACKS}
    license_root = PACKAGE / "licenses"
    if license_root.exists():
        for path in license_root.iterdir():
            if path.is_dir() and path.name not in expected_licenses:
                shutil.rmtree(path)

    manifest = {
        "name": "quanttrio-llmbench-data-all",
        "version": VERSION,
        "package": "llmbench_data_all",
        "source_revision": source_revisions,
        "source_packs": source_packs,
        "datasets": datasets,
    }
    (PACKAGE / "pack.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Wrote {len(datasets)} datasets, {sum(item['count'] for item in datasets.values())} "
        f"records, and {total_bytes} bytes"
    )


if __name__ == "__main__":
    main()
