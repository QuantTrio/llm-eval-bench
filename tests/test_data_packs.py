from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from pathlib import Path

from typer.testing import CliRunner

from llmbench import data_packs
from llmbench.capabilities import BENCHMARK_CAPABILITIES, capability_matrix
from llmbench.cli import app
from llmbench.datasets import list_datasets, load_dataset


class EntryPoint:
    name = "test-pack"

    def __init__(self, manifest) -> None:
        self.manifest = manifest

    def load(self):
        return lambda: self.manifest


def manifest() -> dict:
    payload = files("llmbench").joinpath("data", "stress.jsonl").read_bytes()
    return {
        "name": "test-pack",
        "version": "1.0.0",
        "package": "llmbench",
        "datasets": {
            "external-stress": {
                "file": "data/stress.jsonl",
                "count": 10,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "category": "Performance",
                "metric": "none",
                "type": "stress",
                "license": "Apache-2.0",
                "restriction": None,
                "source": "test fixture",
                "recommended_max_tokens": 128,
            }
        },
    }


def test_data_pack_discovery_loading_verification_and_cli(monkeypatch) -> None:
    monkeypatch.setattr(data_packs, "_entry_points", lambda: [EntryPoint(manifest())])
    packs = data_packs.discover_data_packs()
    assert packs[0]["name"] == "test-pack"
    verified = data_packs.verify_installed_data_packs()
    assert verified == [
        {
            "dataset": "external-stress",
            "pack": "test-pack",
            "version": "1.0.0",
            "sha256": manifest()["datasets"]["external-stress"]["sha256"],
            "sha256_valid": True,
            "count": 10,
            "count_valid": True,
        }
    ]
    assert any(item["name"] == "external-stress" for item in list_datasets())
    assert len(load_dataset("external-stress")) == 10

    listed = CliRunner().invoke(app, ["data", "list"])
    assert listed.exit_code == 0
    assert "test-pack 1.0.0" in listed.stdout
    checked = CliRunner().invoke(app, ["data", "verify"])
    assert checked.exit_code == 0
    assert "status=ok" in checked.stdout


def test_empty_data_packs_and_category_coverage(monkeypatch) -> None:
    monkeypatch.setattr(data_packs, "_entry_points", lambda: [])
    listed = CliRunner().invoke(app, ["data", "list"])
    assert listed.exit_code == 0
    assert "No optional data packs" in listed.stdout
    checked = CliRunner().invoke(app, ["data", "verify"])
    assert checked.exit_code == 0

    rows = capability_matrix({"truthfulqa", "gpqa-diamond", "drop"})
    assert len(rows) == 20 == len(BENCHMARK_CAPABILITIES)
    assert sum(row["installed"] for row in rows) == 3
    coverage = CliRunner().invoke(app, ["coverage"])
    assert coverage.exit_code == 0
    assert "Coverage: 3/20 categories" in coverage.stdout


def test_consolidated_public_pack_manifest() -> None:
    path = (
        Path(__file__).parents[1] / "data-packs" / "all-public" / "llmbench_data_all" / "pack.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "agieval",
        "aider-polyglot",
        "aime-2025",
        "browsecomp",
        "humaneval",
        "ifbench",
        "longbench-v2",
        "mmmu",
        "pinchbench",
        "simple-bench",
        "simpleqa",
        "terminal-bench-2",
    }
    assert payload["name"] == "quanttrio-llmbench-data-all"
    assert payload["version"] == "0.5.0"
    assert set(payload["datasets"]) == expected
    assert set(payload["source_packs"]) == expected
    assert all(item["source_pack_version"] == "0.5.0" for item in payload["datasets"].values())
