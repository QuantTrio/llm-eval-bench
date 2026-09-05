from __future__ import annotations

import pytest

from llmbench.config import load_bench_config, load_yaml, secret_from_env
from llmbench.runspec import RunSpec, SpecError


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


def test_endpoint_precedence_cli_env_yaml_defaults(tmp_path, monkeypatch):
    config = tmp_path / "bench.yaml"
    config.write_text(
        "schema_version: 2\ntargets:\n  chat:\n    provider: anthropic\n"
        "    api: messages\n    base_url: http://yaml.test/prefix\n"
        "    api_key: yaml-key\n    model: exact-yaml-id\n"
    )
    monkeypatch.setenv("LLMBENCH_PROVIDER", "gemini")
    monkeypatch.setenv("LLMBENCH_API", "generate-content")
    monkeypatch.setenv("LLMBENCH_BASE_URL", "http://env.test/prefix")
    monkeypatch.setenv("LLMBENCH_API_KEY", "env-key")
    spec = RunSpec.resolve({}, explicit=set(), config_path=config)
    assert (spec.provider, spec.api, spec.base_url, spec.api_key) == (
        "gemini",
        "generate-content",
        "http://env.test/prefix",
        "env-key",
    )
    explicit = {
        "provider": "xai",
        "api": "responses",
        "base_url": "http://cli.test/prefix",
        "api_key": "cli-key",
        "model": "exact-cli-id",
    }
    spec = RunSpec.resolve(explicit, explicit=set(explicit), config_path=config)
    assert (spec.provider, spec.api, spec.base_url, spec.api_key, spec.model) == (
        "xai",
        "responses",
        "http://cli.test/prefix",
        "cli-key",
        "exact-cli-id",
    )


def test_explicit_provider_uses_its_own_env_key_after_yaml(tmp_path, monkeypatch):
    config = tmp_path / "bench.yaml"
    config.write_text(
        "schema_version: 2\ntargets:\n  chat:\n    provider: anthropic\n"
        "    base_url: http://yaml.test/v1\n    api_key: yaml-key\n    model: test-id\n"
    )
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-vendor")
    monkeypatch.setenv("XAI_API_KEY", "xai-key")
    monkeypatch.delenv("LLMBENCH_API_KEY", raising=False)
    spec = RunSpec.resolve(
        {"provider": "xai", "api": None}, explicit={"provider"}, config_path=config
    )
    assert spec.api_key == "xai-key"
    assert spec.api == "responses"


def _saved_run(tmp_path):
    import json

    spec = RunSpec.resolve(
        {"base_url": "http://localhost:8000/v1", "model": "saved-model", "api": "responses"},
        explicit={"base_url", "model", "api"},
    )
    manifest = {
        "mode": "run",
        "base_url": spec.base_url,
        "model": spec.model,
        "config": spec.to_config(),
    }
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))
    return spec


def test_resume_restores_model_and_endpoint_not_environment(tmp_path, monkeypatch):
    _saved_run(tmp_path)
    monkeypatch.setenv("LLMBENCH_BASE_URL", "http://different.test/v1")
    monkeypatch.setenv("LLMBENCH_API", "messages")
    restored = RunSpec.resolve({}, explicit=set(), resume=tmp_path, mode="run")
    assert restored.model == "saved-model"
    assert restored.api == "responses"
    assert restored.base_url == "http://localhost:8000/v1"


@pytest.mark.parametrize(
    ("key", "value"),
    [("model", "different-id"), ("api", "messages"), ("base_url", "http://different.test/v1")],
)
def test_resume_rejects_explicit_endpoint_or_model_changes(tmp_path, key, value):
    _saved_run(tmp_path)
    with pytest.raises(SpecError, match=key):
        RunSpec.resolve({key: value}, explicit={key}, resume=tmp_path, mode="run")


def test_resume_allows_same_model_and_rotated_credentials(tmp_path):
    _saved_run(tmp_path)
    spec = RunSpec.resolve(
        {"model": "saved-model", "api": "responses", "api_key": "rotated-key"},
        explicit={"model", "api", "api_key"},
        resume=tmp_path,
        mode="run",
    )
    assert spec.model == "saved-model"
    assert spec.api_key == "rotated-key"


def test_legacy_manifest_uses_chat_even_when_current_defaults_differ(tmp_path):
    import json

    _saved_run(tmp_path)
    path = tmp_path / "run_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["config"].pop("api")
    manifest["config"].pop("provider")
    path.write_text(json.dumps(manifest))
    spec = RunSpec.resolve({"api": "responses"}, explicit=set(), resume=tmp_path, mode="run")
    assert spec.api == "chat"


@pytest.mark.parametrize("api", ["responses", "messages", "generate-content"])
def test_nonchat_apis_do_not_force_sampling_parameters(api):
    spec = RunSpec.resolve(
        {"model": "reasoning-model", "base_url": "http://localhost:8000/v1", "api": api},
        explicit={"model", "base_url", "api"},
    )
    assert spec.temperature is None
    assert spec.top_p is None


def test_local_chat_defaults_are_deterministic():
    spec = RunSpec.resolve(
        {"model": "test-model", "base_url": "http://localhost:8000/v1", "api": "chat"},
        explicit={"model", "base_url", "api"},
    )
    assert spec.temperature == 0.0
    assert spec.top_p == 1.0


def test_explicit_key_overrides_an_unset_yaml_key_reference(tmp_path, monkeypatch):
    monkeypatch.delenv("UNSET_YAML_SECRET", raising=False)
    config = tmp_path / "bench.yaml"
    config.write_text(
        "schema_version: 2\ntargets:\n  chat:\n    base_url: http://localhost:8000/v1\n"
        "    model: test-id\n    api_key_env: UNSET_YAML_SECRET\n"
    )
    spec = RunSpec.resolve({"api_key": "EMPTY"}, explicit={"api_key"}, config_path=config)
    assert spec.api_key == ""


def test_resume_rejects_a_changed_run_mode(tmp_path):
    import json

    _saved_run(tmp_path)
    path = tmp_path / "run_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["config"]["run_mode"] = "quality"
    path.write_text(json.dumps(manifest))
    with pytest.raises(SpecError, match="run_mode"):
        RunSpec.resolve({"run_mode": "both"}, explicit={"run_mode"}, resume=tmp_path, mode="run")


def test_resume_restores_yaml_key_env_name_without_persisting_secret(tmp_path, monkeypatch):
    import json

    monkeypatch.delenv("LLMBENCH_API_KEY", raising=False)
    monkeypatch.setenv("CHAT_API_KEY", "first-local-secret")
    config = tmp_path / "bench.yaml"
    config.write_text(
        "schema_version: 2\ntargets:\n  chat:\n    model: local-model\n"
        "    base_url: http://localhost:8000/v1\n    api_key_env: CHAT_API_KEY\n"
    )
    original = RunSpec.resolve({}, explicit=set(), config_path=config)
    assert original.api_key == "first-local-secret"
    recorded = original.to_config()
    assert recorded["api_key_env"] == "CHAT_API_KEY"
    assert "api_key" not in recorded
    manifest = json.dumps(
        {
            "mode": "run",
            "model": original.model,
            "base_url": original.base_url,
            "config": recorded,
        }
    )
    assert "first-local-secret" not in manifest
    (tmp_path / "run_manifest.json").write_text(manifest)
    monkeypatch.setenv("CHAT_API_KEY", "rotated-local-secret")
    resumed = RunSpec.resolve({}, explicit=set(), resume=tmp_path, mode="run")
    assert resumed.api_key == "rotated-local-secret"
    assert resumed.api_key_env == "CHAT_API_KEY"
    assert resumed.to_config() == recorded
    monkeypatch.setenv("LLMBENCH_API_KEY", "environment-override")
    assert RunSpec.resolve({}, explicit=set(), resume=tmp_path).api_key == "environment-override"
    assert (
        RunSpec.resolve({"api_key": "cli-override"}, explicit={"api_key"}, resume=tmp_path).api_key
        == "cli-override"
    )


@pytest.mark.parametrize("field", ["temperature", "top_p"])
def test_yaml_null_sampling_parameter_stays_omitted_on_resume(tmp_path, field):
    import json

    config = tmp_path / "bench.yaml"
    config.write_text(
        "schema_version: 2\ntargets:\n  chat:\n    model: local-model\n"
        f"    base_url: http://localhost:8000/v1\nrun:\n  {field}: null\n"
    )
    original = RunSpec.resolve({}, explicit=set(), config_path=config)
    assert getattr(original, field) is None
    other = "top_p" if field == "temperature" else "temperature"
    assert getattr(original, other) == (1.0 if other == "top_p" else 0.0)
    (tmp_path / "run_manifest.json").write_text(
        json.dumps(
            {
                "mode": "run",
                "model": original.model,
                "base_url": original.base_url,
                "config": original.to_config(),
            }
        )
    )
    resumed = RunSpec.resolve({}, explicit=set(), resume=tmp_path, mode="run")
    assert getattr(resumed, field) is None
    assert resumed.to_config() == original.to_config()
