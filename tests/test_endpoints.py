from __future__ import annotations

import pytest

from llmbench.endpoints import resolve_endpoint


@pytest.fixture(autouse=True)
def clean_endpoint_environment(monkeypatch):
    for name in (
        "LLMBENCH_PROVIDER",
        "LLMBENCH_API",
        "LLMBENCH_BASE_URL",
        "LLMBENCH_API_KEY",
        "OPENAI_API_KEY",
        "XAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
    ("provider", "api", "url", "key_env"),
    [
        ("openai", "responses", "https://api.openai.com/v1", "OPENAI_API_KEY"),
        ("xai", "responses", "https://api.x.ai/v1", "XAI_API_KEY"),
        ("anthropic", "messages", "https://api.anthropic.com/v1", "ANTHROPIC_API_KEY"),
        (
            "gemini",
            "generate-content",
            "https://generativelanguage.googleapis.com/v1beta",
            "GEMINI_API_KEY",
        ),
    ],
)
def test_official_defaults_use_their_own_key(provider, api, url, key_env, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-openai")
    monkeypatch.setenv(key_env, "correct-key")
    endpoint = resolve_endpoint(provider=provider)
    assert (endpoint.provider, endpoint.api, endpoint.base_url, endpoint.api_key) == (
        provider,
        api,
        url,
        "correct-key",
    )


@pytest.mark.parametrize("provider", ["anthropic", "gemini", "xai"])
def test_another_provider_does_not_inherit_openai_key(provider, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-key")
    with pytest.raises(ValueError, match="API key is required"):
        resolve_endpoint(provider=provider)


def test_custom_url_keeps_prefix_and_accepts_key_or_no_auth(monkeypatch):
    monkeypatch.setenv("LLMBENCH_BASE_URL", "https://ignored.test/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    endpoint = resolve_endpoint(base_url="http://localhost:8000/gateway/v1///")
    assert endpoint.base_url == "http://localhost:8000/gateway/v1"
    assert endpoint.api == "chat"
    assert endpoint.api_key == ""
    assert endpoint.provider is None
    assert resolve_endpoint(base_url=endpoint.base_url, api_key="local-key").api_key == "local-key"
    monkeypatch.setenv("LLMBENCH_API_KEY", "generic-key")
    assert resolve_endpoint(base_url=endpoint.base_url, api_key="EMPTY").api_key == ""


@pytest.mark.parametrize("api", ["chat", "responses", "messages", "generate-content"])
def test_custom_url_selects_exact_protocol(api):
    endpoint = resolve_endpoint(base_url="http://localhost:8000/prefix", api=api)
    assert endpoint.api == api
    assert endpoint.base_url == "http://localhost:8000/prefix"


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("anthropic", "messages"),
        ("generateContent", "generate-content"),
        ("gemini", "generate-content"),
    ],
)
def test_protocol_aliases(alias, expected):
    assert resolve_endpoint(base_url="http://localhost", api=alias).api == expected


@pytest.mark.parametrize(
    "url",
    ["localhost:8000", "ftp://localhost/v1", "https://user:pass@host/v1", "https://host/v1?k=v"],
)
def test_invalid_urls_rejected(url):
    with pytest.raises(ValueError, match="base URL"):
        resolve_endpoint(base_url=url)


def test_lookalike_host_is_not_inferred_as_official(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    endpoint = resolve_endpoint(base_url="https://evilapi.openai.com/v1")
    assert endpoint.provider is None
    assert endpoint.api_key == ""


def test_environment_is_below_explicit_endpoint_options(monkeypatch):
    monkeypatch.setenv("LLMBENCH_PROVIDER", "anthropic")
    monkeypatch.setenv("LLMBENCH_API", "messages")
    monkeypatch.setenv("LLMBENCH_BASE_URL", "http://localhost:8080/custom")
    monkeypatch.setenv("LLMBENCH_API_KEY", "env-key")
    default = resolve_endpoint()
    assert (default.provider, default.api, default.base_url, default.api_key) == (
        "anthropic",
        "messages",
        "http://localhost:8080/custom",
        "env-key",
    )
    explicit = resolve_endpoint(
        provider="xai", api="chat", base_url="http://localhost:9000/v1", api_key="cli-key"
    )
    assert (explicit.provider, explicit.api, explicit.base_url, explicit.api_key) == (
        "xai",
        "chat",
        "http://localhost:9000/v1",
        "cli-key",
    )
