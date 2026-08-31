#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    with zipfile.ZipFile(args.wheel) as wheel:
        names = set(wheel.namelist())
        manifest_name = next(name for name in names if name.endswith("llmbench/data/manifest.json"))
        catalog_name = next(
            name for name in names if name.endswith("llmbench/data/benchmark_catalog.json")
        )
        manifest = json.loads(wheel.read(manifest_name))
        catalog = json.loads(wheel.read(catalog_name))
        for metadata in manifest.values():
            suffix = f"llmbench/data/{metadata['file']}"
            assert any(name.endswith(suffix) for name in names), suffix
        assert len(manifest) == 9
        assert catalog["count"] == 155
    print(f"Verified {args.wheel}: 9 bundled resources, 155 catalog entries")


if __name__ == "__main__":
    main()
