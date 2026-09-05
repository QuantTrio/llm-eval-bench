"""Resolve endpoint defaults without probing or silently changing protocols."""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlparse, urlunparse

from .protocols import SUPPORTED_APIS

_PROVIDERS = {
    "openai": ("https://api.openai.com/v1", "responses", "OPENAI_API_KEY"),
    "xai": ("https://api.x.ai/v1", "responses", "XAI_API_KEY"),
    "anthropic": ("https://api.anthropic.com/v1", "messages", "ANTHROPIC_API_KEY"),
    "gemini": (
        "https://generativelanguage.googleapis.com/v1beta",
        "generate-content",
        "GEMINI_API_KEY",
    ),
}


@dataclass(frozen=True, slots=True)
class Endpoint:
    provider: str | None
    api: str
    base_url: str
    api_key: str


def _env(name: str) -> str | None:
    return os.environ.get(name, "").strip() or None


def normalize_provider(value: str | None) -> str | None:
    value = (value or "").strip().lower()
    value = {"grok": "xai", "claude": "anthropic", "google": "gemini"}.get(value, value)
    if not value:
        return None
    if value not in _PROVIDERS:
        raise ValueError(f"unsupported provider '{value}'; choose {', '.join(_PROVIDERS)}")
    return value


def normalize_api(value: str | None) -> str | None:
    value = (value or "").strip().lower().replace("_", "-")
    value = {
        "anthropic": "messages",
        "gemini": "generate-content",
        "generatecontent": "generate-content",
        "chat-completions": "chat",
    }.get(value, value)
    if not value:
        return None
    if value not in SUPPORTED_APIS:
        raise ValueError(f"unsupported api '{value}'; choose {', '.join(SUPPORTED_APIS)}")
    return value


def infer_provider(base_url: str) -> str | None:
    host = (urlparse(base_url).hostname or "").lower()
    return next(
        (name for name, (url, _, _) in _PROVIDERS.items() if host == urlparse(url).hostname),
        None,
    )


def endpoint_environment(provider: str | None, base_url: str | None) -> dict[str, str]:
    """Environment values to overlay above YAML, below explicit CLI options."""
    values = {
        key: value
        for key, env in (
            ("provider", "LLMBENCH_PROVIDER"),
            ("api", "LLMBENCH_API"),
            ("base_url", "LLMBENCH_BASE_URL"),
            ("api_key", "LLMBENCH_API_KEY"),
        )
        if (value := _env(env)) is not None
    }
    selected = normalize_provider(provider) or infer_provider(base_url or "")
    if selected is None and not base_url:
        selected = "openai"
    if "api_key" not in values and selected is not None:
        key = _env(_PROVIDERS[selected][2])
        if selected == "gemini" and key is None:
            key = _env("GOOGLE_API_KEY")
        if key is not None:
            values["api_key"] = key
    return values


def normalize_base_url(raw: str, *, api: str) -> str:
    parsed = urlparse(raw.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("base URL must be an absolute http(s) URL, e.g. http://localhost:8000/v1")
    if parsed.username or parsed.password:
        raise ValueError("base URL credentials are not allowed; use --api-key")
    if parsed.query or parsed.fragment or parsed.params:
        raise ValueError("base URL must not include query, fragment, or URL parameters")
    path = parsed.path.rstrip("/")
    if not path:
        path = "/v1beta" if api == "generate-content" else "/v1"
    return urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))


def resolve_endpoint(
    *,
    provider: str | None = None,
    api: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    use_environment: bool = True,
) -> Endpoint:
    if use_environment:
        provider = provider if provider is not None else _env("LLMBENCH_PROVIDER")
        api = api if api is not None else _env("LLMBENCH_API")
        base_url = base_url or _env("LLMBENCH_BASE_URL")
    provider = normalize_provider(provider) or infer_provider(base_url or "")
    if not base_url:
        provider = provider or "openai"
        base_url = _PROVIDERS[provider][0]
    api = normalize_api(api) or (_PROVIDERS[provider][1] if provider else "chat")
    base_url = normalize_base_url(base_url, api=api)
    if use_environment and api_key is None:
        api_key = endpoint_environment(provider, base_url).get("api_key")
    key = (api_key or "").strip()
    if key.upper() == "EMPTY":
        key = ""
    official_provider = infer_provider(base_url)
    if official_provider is not None and not key:
        raise ValueError(
            f"API key is required for {official_provider}; set {_PROVIDERS[official_provider][2]} "
            "or --api-key"
        )
    return Endpoint(provider=provider, api=api, base_url=base_url, api_key=key)
