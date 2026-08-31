from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

VERSION = "1.0.1"
ROOT = Path(__file__).resolve().parents[2]
BUNDLE = Path(__file__).parent
PACKAGES = BUNDLE / "packages"


def _copy_package(source: Path, destination: Path) -> tuple[int, int]:
    files = 0
    total_bytes = 0
    for path in sorted(source.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        files += 1
        total_bytes += path.stat().st_size
    return files, total_bytes


def main() -> None:
    core_source = ROOT / "src" / "llmbench"
    data_source = ROOT / "data-packs" / "all-public" / "llmbench_data_all"
    core_version_text = (core_source / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*"([^"]+)"', core_version_text)
    if not match or match.group(1) != VERSION:
        raise ValueError(f"core version does not match bundle version {VERSION}")
    manifest_path = data_source / "pack.json"
    if not manifest_path.exists():
        raise FileNotFoundError("run data-packs/all-public/build_pack.py before bundling")
    data_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, metadata in data_manifest["datasets"].items():
        path = data_source / metadata["file"]
        if not path.exists():
            raise FileNotFoundError(f"aggregate data asset is missing: {name}: {path}")

    PACKAGES.mkdir(parents=True, exist_ok=True)
    core_files, core_bytes = _copy_package(core_source, PACKAGES / "llmbench")
    data_files, data_bytes = _copy_package(data_source, PACKAGES / "llmbench_data_all")
    shutil.copy2(ROOT / "LICENSE", BUNDLE / "LICENSE")
    print(
        f"Staged {core_files} framework files ({core_bytes} bytes) and "
        f"{data_files} data files ({data_bytes} bytes)"
    )


if __name__ == "__main__":
    main()
