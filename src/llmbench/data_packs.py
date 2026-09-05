from __future__ import annotations

import hashlib
import json
from collections.abc import Collection
from dataclasses import dataclass
from importlib import metadata, resources
from typing import Any


@dataclass(frozen=True, slots=True)
class DatasetResource:
    name: str
    package: str
    file: str
    metadata: dict[str, Any]
    pack: str
    pack_version: str

    def read_bytes(self) -> bytes:
        return resources.files(self.package).joinpath(self.file).read_bytes()

    def open_text(self):
        return resources.files(self.package).joinpath(self.file).open("r", encoding="utf-8")


def _entry_points():
    discovered = metadata.entry_points()
    if hasattr(discovered, "select"):
        return discovered.select(group="llmbench.data_packs")
    return discovered.get("llmbench.data_packs", [])


def discover_data_packs() -> list[dict[str, Any]]:
    packs = []
    for entry_point in _entry_points():
        factory = entry_point.load()
        payload = factory() if callable(factory) else factory
        if not isinstance(payload, dict):
            raise ValueError(f"Data pack {entry_point.name!r} did not return a manifest object")
        required = {"name", "version", "package", "datasets"}
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"Data pack {entry_point.name!r} is missing fields: {missing}")
        if not isinstance(payload["datasets"], dict):
            raise ValueError(f"Data pack {entry_point.name!r} datasets must be an object")
        packs.append(payload)
    return sorted(packs, key=lambda item: (item["name"], item["version"]))


def external_dataset_resources(
    *, exclude_names: Collection[str] = ()
) -> dict[str, DatasetResource]:
    """Discover optional datasets, ignoring IDs already owned by the core wheel.

    Filtering precedes duplicate detection so multiple legacy packs cannot shadow
    a newly bundled dataset. Other external duplicates still fail explicitly.
    """
    datasets: dict[str, DatasetResource] = {}
    for pack in discover_data_packs():
        for name, item in pack["datasets"].items():
            if name in exclude_names:
                continue
            if name in datasets:
                raise ValueError(f"Dataset {name!r} is provided by multiple installed data packs")
            if not isinstance(item, dict) or "file" not in item:
                raise ValueError(f"Dataset {name!r} in pack {pack['name']!r} has no file")
            datasets[name] = DatasetResource(
                name=name,
                package=str(pack["package"]),
                file=str(item["file"]),
                metadata=dict(item),
                pack=str(pack["name"]),
                pack_version=str(pack["version"]),
            )
    return datasets


def verify_installed_data_packs() -> list[dict[str, Any]]:
    verified = []
    for name, resource in external_dataset_resources().items():
        payload = resource.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        expected = resource.metadata.get("sha256")
        lines = sum(bool(line.strip()) for line in payload.splitlines())
        expected_count = resource.metadata.get("count")
        verified.append(
            {
                "dataset": name,
                "pack": resource.pack,
                "version": resource.pack_version,
                "sha256": digest,
                "sha256_valid": expected is None or digest == expected,
                "count": lines,
                "count_valid": expected_count is None or lines == expected_count,
            }
        )
    return sorted(verified, key=lambda item: item["dataset"])


def manifest_from_package(package: str, path: str = "pack.json") -> dict[str, Any]:
    return json.loads(resources.files(package).joinpath(path).read_text(encoding="utf-8"))
