"""The single resolved description of one benchmark run.

Three sources can describe a run: command-line options, a YAML bench config, and the
manifest of an interrupted run. `RunSpec.resolve` applies them in that order of
precedence, so no other module has to know that more than one source exists.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .config import load_bench_config, secret_from_env
from .endpoints import (
    endpoint_environment,
    infer_provider,
    normalize_api,
    normalize_base_url,
    normalize_provider,
    resolve_endpoint,
)
from .protocols import validate_extra_body

DEFAULT_DATASETS = (
    "mmlu-pro",
    "mmlu-redux",
    "gpqa-diamond",
    "gsm8k",
    "ceval",
    "hellaswag",
    "truthfulqa",
    "drop",
)
DEFAULT_MAX_TOKENS = 4096

# YAML `run:` keys that are spelled differently from the spec attribute they set.
_RUN_ALIASES = {"datasets": "dataset", "limit_per_dataset": "limit"}
# A recorded manifest uses the same spellings, read back the other way round.
_MANIFEST_KEYS = {attribute: key for key, attribute in _RUN_ALIASES.items()}
_PROTECTED_BODY_KEYS = {"model", "messages", "stream"}
PRESETS = {
    "quick": (("mmlu-pro", "gsm8k", "truthfulqa"), 5),
    "standard": (("mmlu-pro", "gpqa-diamond", "gsm8k", "drop", "truthfulqa", "hellaswag"), 100),
    "full": (("mmlu-pro", "gpqa-diamond", "gsm8k", "drop", "truthfulqa", "hellaswag"), None),
}
# Everything a resumed run must repeat exactly, rather than re-read from the CLI.
_RESUMED = (
    "dataset",
    "limit",
    "sample",
    "concurrency",
    "temperature",
    "top_p",
    "max_tokens",
    "timeout",
    "retries",
    "retry_backoff",
    "n_samples",
    "seed",
    "stream",
    "memory_gb",
    "checkpoint_every",
    "progress_interval",
    "request_extra_body",
    "api",
    "provider",
    "preset",
    "run_mode",
    "api_key_env",
)


class SpecError(ValueError):
    """A run description that cannot be resolved into a usable RunSpec."""


@dataclass(slots=True)
class RunSpec:
    base_url: str = ""
    api_key: str | None = None
    api_key_env: str | None = None
    provider: str | None = None
    api: str | None = "chat"
    model: str | None = None
    dataset: tuple[str, ...] = DEFAULT_DATASETS
    limit: int | None = 100
    preset: str | None = None
    run_mode: str | None = None
    sample: int | None = None
    concurrency: int = 1
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    timeout: float = 120.0
    retries: int = 2
    retry_backoff: float = 2.0
    n_samples: int = 1
    seed: int = 42
    stream: bool = False
    output_dir: Path | None = None
    memory_gb: float | None = None
    checkpoint_every: int = 1
    progress_interval: float = 5.0
    request_extra_body: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def resolve(
        cls,
        values: dict[str, Any],
        *,
        explicit: set[str],
        config_path: Path | None = None,
        resume: Path | None = None,
        mode: str | None = None,
    ) -> RunSpec:
        """Build a spec from CLI `values`, overlaying YAML then a resumed manifest.

        `explicit` names the options the user actually typed; anything else may be
        replaced by the YAML config. A resume replaces the whole run description,
        because a resumed run must repeat the original one exactly.
        """
        spec = cls(**{key: _coerce(key, value) for key, value in values.items()})
        configured: dict[str, Any] = {}
        if config_path is not None and resume is None:
            configured = _read_config(config_path)
            spec = spec._overlay(configured, skip=explicit)
        if resume is not None:
            if config_path is not None:
                raise SpecError("--config cannot be combined with --resume")
            restored = spec._resume_from(resume, mode=mode)
            for key in explicit & (set(_RESUMED) | {"model", "base_url", "output_dir"}):
                actual, expected = getattr(spec, key), getattr(restored, key)
                if key == "base_url":
                    actual = normalize_base_url(actual, api=restored.api or "chat")
                elif key == "api":
                    actual = normalize_api(actual)
                elif key == "provider":
                    actual = normalize_provider(actual)
                if actual != expected:
                    raise SpecError(f"cannot override resumed run field '{key}'; start a new run")
            spec = restored
        if not isinstance(spec.model, str) or not spec.model.strip():
            raise SpecError("--model MODEL_ID is required (or set targets.chat.model in YAML)")
        spec = replace(spec, model=spec.model.strip())
        try:
            if resume is None:
                environment = endpoint_environment(spec.provider, spec.base_url)
                spec = spec._overlay(environment, skip=explicit | {"api_key"})
            # Choose the key after resolving provider and URL, so another vendor's key
            # can never be inherited just because OPENAI_API_KEY happens to be set.
            environment = endpoint_environment(spec.provider, spec.base_url)
            if "api_key" not in explicit and environment.get("api_key") is not None:
                spec = replace(spec, api_key=environment["api_key"])
            elif "api_key" not in explicit and spec.api_key_env is not None:
                spec = replace(spec, api_key=secret_from_env(spec.api_key_env))
            endpoint = resolve_endpoint(
                provider=spec.provider,
                api=spec.api,
                base_url=spec.base_url,
                api_key=spec.api_key,
                use_environment=False,
            )
            validate_extra_body(endpoint.api, spec.request_extra_body)
        except ValueError as exc:
            raise SpecError(str(exc)) from exc
        spec = replace(
            spec,
            provider=endpoint.provider,
            api=endpoint.api,
            base_url=endpoint.base_url,
            api_key=endpoint.api_key,
        )
        if resume is None and spec.api == "chat" and infer_provider(spec.base_url) is None:
            supplied = explicit | set(configured)
            spec = replace(
                spec,
                **{
                    key: default
                    for key, default in (("temperature", 0.0), ("top_p", 1.0))
                    if getattr(spec, key) is None and key not in supplied
                },
            )
        if resume is None and spec.preset is not None:
            if spec.preset not in PRESETS:
                raise SpecError(f"unknown preset '{spec.preset}'")
            datasets, count = PRESETS[spec.preset]
            supplied = explicit | set(configured)
            changes: dict[str, Any] = {}
            if "dataset" not in supplied:
                changes["dataset"] = datasets
            if not ({"limit", "sample"} & supplied):
                changes.update(limit=count, sample=None)
            spec = replace(spec, **changes)
        if not spec.dataset:
            raise SpecError("at least one dataset is required")
        return spec

    def _overlay(self, configured: dict[str, Any], *, skip: set[str]) -> RunSpec:
        supplied = {
            key: _coerce(key, value) for key, value in configured.items() if key not in skip
        }
        return replace(self, **supplied) if supplied else self

    def _resume_from(self, directory: Path, *, mode: str | None) -> RunSpec:
        path = directory / "run_manifest.json"
        if not path.exists():
            raise SpecError(f"resume manifest not found: {path}")
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SpecError(f"cannot read resume manifest: {exc}") from exc
        if mode is not None and manifest.get("mode") != mode:
            raise SpecError(f"cannot resume mode {manifest.get('mode')!r} with command {mode!r}")
        if not isinstance(manifest.get("config"), dict):
            raise SpecError("resume manifest must contain a config object")
        stored = manifest["config"]
        carried = {
            attribute: stored[_MANIFEST_KEYS.get(attribute, attribute)]
            for attribute in _RESUMED
            if _MANIFEST_KEYS.get(attribute, attribute) in stored
        }
        carried["api"] = stored.get("api", "chat")
        carried["provider"] = stored.get("provider")
        carried["preset"] = stored.get("preset")
        carried["run_mode"] = stored.get("run_mode")
        carried["base_url"] = manifest.get("base_url", "")
        carried["model"] = manifest.get("model")
        carried["output_dir"] = directory
        return replace(self, **{key: _coerce(key, value) for key, value in carried.items()})

    def to_config(
        self,
        *,
        dataset_max_tokens: dict[str, int] | None = None,
        available_models: list[str] | None = None,
    ) -> dict[str, Any]:
        """The run description recorded in the manifest and summary."""
        config: dict[str, Any] = {
            "api": self.api,
            "provider": self.provider,
            "preset": self.preset,
            "run_mode": self.run_mode,
            "datasets": list(self.dataset),
            "limit_per_dataset": self.limit,
            "sample": self.sample,
            "concurrency": self.concurrency,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "retries": self.retries,
            "retry_backoff": self.retry_backoff,
            "n_samples": self.n_samples,
            "seed": self.seed,
            "stream": self.stream,
            "memory_gb": self.memory_gb,
            "request_extra_body": self.request_extra_body,
            "checkpoint_every": self.checkpoint_every,
            "progress_interval": self.progress_interval,
        }
        if dataset_max_tokens is not None:
            config["default_max_tokens"] = DEFAULT_MAX_TOKENS
            config["dataset_max_tokens"] = dataset_max_tokens
        if available_models is not None:
            config["available_models"] = available_models
        if self.api_key_env is not None:
            config["api_key_env"] = self.api_key_env
        return config

    def resolved_max_tokens(self, recommended: int) -> int:
        if self.max_tokens is not None:
            return self.max_tokens
        return max(DEFAULT_MAX_TOKENS, recommended)


@dataclass(slots=True)
class LoadSpec:
    """The load-generation knobs that only the performance modes use."""

    levels: tuple[int, ...] = (64,)
    duration: float = 60.0
    requests: int | None = None
    warmup_requests: int = 0
    request_rate: float | None = None
    ramp_seconds: float = 0.0
    prompt_profile: str = "mixed"
    server_metrics: bool = True

    @classmethod
    def from_cli(cls, concurrency: str, **values: Any) -> LoadSpec:
        try:
            levels = tuple(int(part.strip()) for part in concurrency.split(",") if part.strip())
        except ValueError as exc:
            raise SpecError("--concurrency must be comma-separated integers") from exc
        if not levels or any(level < 1 for level in levels):
            raise SpecError("--concurrency values must all be at least 1")
        return cls(levels=levels, **values)

    @property
    def is_sweep(self) -> bool:
        return len(self.levels) > 1

    def to_config(self, level: int, spec: RunSpec, available_models: list[str]) -> dict[str, Any]:
        return {
            "api": spec.api,
            "provider": spec.provider,
            "run_mode": "load",
            "temperature": spec.temperature,
            "top_p": spec.top_p,
            "request_extra_body": spec.request_extra_body,
            "datasets": ["stress"],
            "concurrency": level,
            "duration": self.duration,
            "max_requests": self.requests,
            "warmup_requests": self.warmup_requests,
            "request_rate": self.request_rate,
            "ramp_seconds": self.ramp_seconds,
            "prompt_profile": self.prompt_profile,
            "server_metrics": self.server_metrics,
            "max_tokens": spec.max_tokens,
            "timeout": spec.timeout,
            "retries": spec.retries,
            "retry_backoff": spec.retry_backoff,
            "seed": spec.seed,
            "stream": spec.stream,
            "available_models": available_models,
            "checkpoint_every": spec.checkpoint_every,
            "progress_interval": spec.progress_interval,
        }


def _read_config(path: Path) -> dict[str, Any]:
    """Flatten a bench YAML into RunSpec attribute names."""
    try:
        payload = load_bench_config(path)
    except (OSError, ValueError) as exc:
        raise SpecError(str(exc)) from exc
    target = (payload.get("targets") or {}).get("chat") or {}
    if not isinstance(target, dict):
        raise SpecError("targets.chat must be an object")
    values: dict[str, Any] = {
        key: target.get(key)
        for key in ("base_url", "model", "provider", "api")
        if target.get(key) is not None
    }
    if target.get("api_key") is not None:
        values["api_key"] = target.get("api_key")
    if target.get("api_key_env") is not None:
        values["api_key_env"] = str(target["api_key_env"])
    run = payload.get("run") or {}
    for key, value in run.items():
        attribute = _RUN_ALIASES.get(key, key)
        if attribute in RunSpec.__slots__:
            values[attribute] = value
    return values


def _as_datasets(value: Any) -> tuple[str, ...]:
    names = value.split(",") if isinstance(value, str) else value
    return tuple(str(name).strip() for name in names if str(name).strip())


def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _as_provider(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def _as_extra_body(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise SpecError(f"--request-extra-body must be valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise SpecError("--request-extra-body must be a JSON object")
    protected = _PROTECTED_BODY_KEYS & set(value)
    if protected:
        raise SpecError("--request-extra-body cannot override: " + ", ".join(sorted(protected)))
    return value


# Only the fields whose external spelling differs from their in-memory form.
_COERCERS = {
    "base_url": lambda value: value or "",
    "api_key": lambda value: value,
    "provider": _as_provider,
    "api": normalize_api,
    "dataset": lambda value: DEFAULT_DATASETS if value is None else _as_datasets(value),
    "temperature": _as_optional_float,
    "top_p": _as_optional_float,
    "output_dir": lambda value: None if value is None else Path(value),
    "request_extra_body": _as_extra_body,
}


def _coerce(key: str, value: Any) -> Any:
    coerce = _COERCERS.get(key)
    return coerce(value) if coerce else value
