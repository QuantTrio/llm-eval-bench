"""The pack declarations must reproduce the packs already on disk.

Building a pack needs the network; checking one does not. Every shipped record is fed
back through its own `Pack` declaration and must come out byte-identical, because
`pack.json` carries a SHA256 of the JSONL that `llmbench data verify` enforces.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

PACKS_ROOT = Path(__file__).resolve().parents[1] / "data-packs"
pytestmark = pytest.mark.skipif(
    not (PACKS_ROOT / "packbuild.py").exists(),
    reason="data-packs/ is a source-tree-only maintainer directory",
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolve their annotations through sys.modules, so register first.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _builders() -> list[Path]:
    return sorted(
        path
        for path in PACKS_ROOT.glob("*/build_pack.py")
        if path.parent.name != "all-public"  # an aggregator, not a source pack
    )


def _split_metadata(metadata: dict, pack) -> tuple[dict, dict]:
    """Recover the per-row metadata a declaration expects, from a shipped record."""
    keys = list(metadata)
    lead = keys[: keys.index("benchmark_category")]
    constant = {"benchmark_category", "benchmark_metric", "recommended_max_tokens"}
    # A `...` placeholder is positional only; its value still comes from the row.
    constant |= {key for key, value in pack.item_metadata.items() if value is not ...}
    constant |= set(pack.flags)
    late = [key for key in keys[len(lead) :] if key not in constant]
    return (
        {key: metadata[key] for key in lead},
        {key: metadata[key] for key in late},
    )


@pytest.mark.parametrize("builder", _builders(), ids=lambda path: path.parent.name)
def test_declaration_reproduces_shipped_pack(builder: Path) -> None:
    packbuild = _load(PACKS_ROOT / "packbuild.py", "packbuild")
    module = _load(builder, f"build_{builder.parent.name.replace('-', '_')}")
    pack = module.PACK

    directory = builder.parent / pack.package
    source = directory / pack.filename
    if not source.is_file():
        pytest.skip(f"optional maintainer source data is not included: {pack.dataset}")
    payload = source.read_bytes()
    shipped = json.loads((directory / "pack.json").read_text(encoding="utf-8"))

    # Split on bytes: some records embed raw U+2028/U+0085, which str.splitlines() treats
    # as line breaks and json.loads() would then choke on.
    records = [json.loads(line) for line in payload.split(b"\n") if line.strip()]
    rebuilt = []
    for record in records:
        metadata, late = _split_metadata(record["metadata"], pack)
        rebuilt.append(
            pack.record(
                id=record["id"],
                question=record["question"],
                answer=record["answer"],
                choices=record.get("choices"),
                subset=record.get("subset"),
                type=record["type"],
                metadata=metadata,
                late_metadata=late,
            )
        )

    assert packbuild.serialize(rebuilt) == payload, f"{pack.dataset}: records changed shape"
    assert pack.manifest(records, payload) == shipped, f"{pack.dataset}: manifest changed shape"


def test_every_declared_pack_has_a_directory() -> None:
    """A declaration naming a package that does not exist would fail only at build time."""
    for builder in _builders():
        pack = _load(builder, f"check_{builder.parent.name.replace('-', '_')}").PACK
        assert (builder.parent / pack.package).is_dir(), pack.package
        assert pack.dataset == builder.parent.name
