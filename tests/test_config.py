from __future__ import annotations

import pytest

from llmbench.config import load_bench_config, load_yaml, secret_from_env


def test_yaml_validation_and_secret_lookup(tmp_path, monkeypatch) -> None:
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    assert load_yaml(empty) == {}

    scalar = tmp_path / "scalar.yaml"
    scalar.write_text("value", encoding="utf-8")
    with pytest.raises(ValueError, match="root"):
        load_yaml(scalar)

    malformed = tmp_path / "malformed.yaml"
    malformed.write_text("value: [", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid YAML"):
        load_yaml(malformed)

    wrong_schema = tmp_path / "wrong-schema.yaml"
    wrong_schema.write_text("schema_version: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        load_bench_config(wrong_schema)

    bad_targets = tmp_path / "bad-targets.yaml"
    bad_targets.write_text("schema_version: 2\ntargets: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="targets"):
        load_bench_config(bad_targets)

    bad_run = tmp_path / "bad-run.yaml"
    bad_run.write_text("schema_version: 2\nrun: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="run"):
        load_bench_config(bad_run)

    monkeypatch.delenv("MISSING_TEST_SECRET", raising=False)
    with pytest.raises(ValueError, match="MISSING_TEST_SECRET"):
        secret_from_env("MISSING_TEST_SECRET")
    monkeypatch.setenv("PRESENT_TEST_SECRET", "secret")
    assert secret_from_env("PRESENT_TEST_SECRET") == "secret"
    assert secret_from_env(None) is None
