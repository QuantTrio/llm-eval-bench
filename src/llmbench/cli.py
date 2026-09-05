"""Command surface only.

Every command does the same three things: collect options, build a spec, call one
module function. Networking, concurrency and file writing live behind those functions.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated, Any

import httpx
import typer

from . import __version__
from .catalog import (
    capability_matrix,
    installed_datasets,
    list_benchmarks,
    report_count_for_dataset,
)
from .client import discover_models
from .comparison import IncomparableRunsError, compare_run_directories, evaluate_policy
from .config import load_yaml
from .data_packs import discover_data_packs, verify_installed_data_packs
from .endpoints import resolve_endpoint
from .executor_client import SmokeFailed
from .report import write_comparison
from .runner import run_evaluation, run_stress
from .runspec import LoadSpec, RunSpec, SpecError
from .session import ResumeMismatch
from .suite import run_suite
from .validation import validate_run_directory

app = typer.Typer(
    name="llmbench",
    help="Independent model evaluation for hosted and local APIs.",
    no_args_is_help=True,
)
executor_app = typer.Typer(help="Serve and manage the remote isolated task executor.")
app.add_typer(executor_app, name="executor", hidden=True)
data_app = typer.Typer(help="Inspect and verify optional benchmark data wheels.")
app.add_typer(data_app, name="data", hidden=True)

# Shared option types: `eval` and `run` differ only in their concurrency and stream
# defaults, so every other option is declared exactly once.
BaseUrl = Annotated[
    str | None,
    typer.Option("--base-url", help="Custom API root; preserves gateway path prefixes."),
]
ApiKey = Annotated[
    str | None,
    typer.Option("--api-key", help="API key; use EMPTY if allowed."),
]
Provider = Annotated[
    str | None, typer.Option("--provider", help="openai, anthropic, xai, or gemini.")
]
Api = Annotated[
    str | None, typer.Option("--api", help="chat, responses, messages, or generate-content.")
]
Model = Annotated[str | None, typer.Option("--model", help="Required model ID; no auto-selection.")]
Dataset = Annotated[str | None, typer.Option("--dataset", "--datasets")]
Limit = Annotated[
    int | None,
    typer.Option(
        "--limit", min=1, help="Per dataset; preset defaults: quick 5, standard 100, full all."
    ),
]
Sample = Annotated[int | None, typer.Option("--sample", min=1)]
Concurrency = Annotated[int | None, typer.Option("--concurrency", min=1)]
Temperature = Annotated[float | None, typer.Option("--temperature", min=0)]
TopP = Annotated[float | None, typer.Option("--top-p", min=0, max=1)]
MaxTokens = Annotated[
    int | None,
    typer.Option("--max-tokens", min=1, help="Defaults to 4096 or a dataset recommendation."),
]
Timeout = Annotated[float, typer.Option("--timeout", min=0.1)]
Retries = Annotated[int, typer.Option("--retries", min=0)]
RetryBackoff = Annotated[float, typer.Option("--retry-backoff", min=0)]
NSamples = Annotated[int, typer.Option("--n-samples", min=1)]
Seed = Annotated[int, typer.Option("--seed")]
Stream = Annotated[bool, typer.Option("--stream/--no-stream")]
OutputDir = Annotated[Path | None, typer.Option("--output-dir")]
MemoryGb = Annotated[float | None, typer.Option("--memory-gb", min=0, hidden=True)]
CheckpointEvery = Annotated[int, typer.Option("--checkpoint-every", min=1)]
ProgressInterval = Annotated[float, typer.Option("--progress-interval", min=0.1)]
Resume = Annotated[Path | None, typer.Option("--resume", exists=True, file_okay=False)]
ExtraBody = Annotated[str | None, typer.Option("--request-extra-body")]
Config = Annotated[Path | None, typer.Option("--config", exists=True, readable=True)]
RequestRate = Annotated[float | None, typer.Option("--request-rate", min=0.01)]
RampSeconds = Annotated[float, typer.Option("--ramp-seconds", min=0)]
PromptProfile = Annotated[str, typer.Option("--prompt-profile")]
ServerMetrics = Annotated[bool, typer.Option("--server-metrics/--no-server-metrics")]


class RunMode(str, Enum):
    quality = "quality"
    load = "load"
    both = "both"


class RunPreset(str, Enum):
    quick = "quick"
    standard = "standard"
    full = "full"


Mode = Annotated[RunMode, typer.Option("--mode")]
Preset = Annotated[RunPreset, typer.Option("--preset")]


def _version(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    """Run offline-ready LLM quality and performance benchmarks."""


def _spec(ctx: typer.Context, values: dict[str, Any], **resolution: Any) -> RunSpec:
    """Resolve CLI options against YAML and resume, keeping typed options authoritative."""
    explicit = {
        name
        for name in values
        if (source := ctx.get_parameter_source(name)) and source.name != "DEFAULT"
    }
    mode_source = ctx.get_parameter_source("mode")
    if "run_mode" in values and mode_source and mode_source.name != "DEFAULT":
        explicit.add("run_mode")
    try:
        return RunSpec.resolve(values, explicit=explicit, **resolution)
    except (OSError, ValueError, TypeError) as exc:
        raise typer.BadParameter(str(exc)) from exc


def _report(summary: dict[str, Any], paths: dict[str, Path]) -> None:
    quality = summary["quality"]
    score = quality.get("composite_score")
    if score is None:
        score = quality.get("sample_mean_score", quality.get("mean_score"))
    performance = summary["performance"]
    typer.echo(f"Model: {summary['model']}")
    typer.echo(
        f"Requests: {performance['total_requests']}  "
        f"Successful: {performance['successful_requests']}  "
        f"Score: {'n/a' if score is None else f'{score:.4f}'}"
    )
    for label in ("summary", "raw", "markdown", "html", "sweep_html"):
        if label in paths:
            typer.echo(f"{label.replace('_', ' ').title()}: {paths[label]}")
    if performance.get("failed_requests", 0) or performance.get("unsupported_requests", 0):
        typer.echo(
            "Evaluation contains failed or unsupported requests; inspect the report.", err=True
        )
        raise typer.Exit(code=4)


def _run_quality(
    ctx: typer.Context,
    values: dict[str, Any],
    *,
    config_path: Path | None,
    resume: Path | None,
    mode: str,
) -> tuple[dict[str, Any], dict[str, Path]]:
    spec = _spec(ctx, values, config_path=config_path, resume=resume, mode=mode)
    try:
        return run_evaluation(spec, mode=mode, resume=resume is not None)
    except (ResumeMismatch, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    except (httpx.HTTPError, OSError) as exc:
        typer.echo(f"Evaluation infrastructure error: {exc}", err=True)
        raise typer.Exit(code=4) from exc


def _run_stress_mode(
    ctx: typer.Context,
    values: dict[str, Any],
    *,
    concurrency: str,
    duration: float,
    requests: int | None,
    warmup_requests: int,
    request_rate: float | None,
    ramp_seconds: float,
    prompt_profile: str,
    server_metrics: bool,
) -> tuple[dict[str, Any], dict[str, Path]]:
    config = values.pop("config", None)
    resume = values.pop("resume", None)
    if resume is not None:
        raise typer.BadParameter("load runs cannot be resumed; choose a new --output-dir")
    spec = _spec(ctx, values, config_path=config, mode="stress")
    try:
        load = LoadSpec.from_cli(
            concurrency or str(spec.concurrency),
            duration=duration,
            requests=requests,
            warmup_requests=warmup_requests,
            request_rate=request_rate,
            ramp_seconds=ramp_seconds,
            prompt_profile=prompt_profile,
            server_metrics=server_metrics,
        )
        return run_stress(spec, load)
    except (SpecError, ValueError, ResumeMismatch) as exc:
        raise typer.BadParameter(str(exc)) from exc
    except (httpx.HTTPError, OSError) as exc:
        typer.echo(f"Evaluation infrastructure error: {exc}", err=True)
        raise typer.Exit(code=4) from exc


@app.command("eval", hidden=True)
def eval_command(
    ctx: typer.Context,
    base_url: BaseUrl = None,
    api_key: ApiKey = None,
    provider: Provider = None,
    api: Api = None,
    model: Model = None,
    dataset: Dataset = None,
    limit: Limit = 100,
    sample: Sample = None,
    concurrency: Concurrency = 1,
    temperature: Temperature = None,
    top_p: TopP = None,
    max_tokens: MaxTokens = None,
    timeout: Timeout = 120.0,
    retries: Retries = 2,
    retry_backoff: RetryBackoff = 2.0,
    n_samples: NSamples = 1,
    seed: Seed = 42,
    stream: Stream = False,
    output_dir: OutputDir = None,
    memory_gb: MemoryGb = None,
    checkpoint_every: CheckpointEvery = 1,
    progress_interval: ProgressInterval = 5.0,
    resume: Resume = None,
    request_extra_body: ExtraBody = None,
    config: Config = None,
) -> None:
    """Run a low-concurrency quality evaluation."""
    _evaluate(ctx, "eval", locals())


@app.command("run")
def run_command(
    ctx: typer.Context,
    base_url: BaseUrl = None,
    api_key: ApiKey = None,
    provider: Provider = None,
    api: Api = None,
    model: Model = None,
    dataset: Dataset = None,
    limit: Limit = None,
    sample: Sample = None,
    concurrency: Concurrency = None,
    temperature: Temperature = None,
    top_p: TopP = None,
    mode: Mode = RunMode.both,
    preset: Preset = RunPreset.quick,
    max_tokens: MaxTokens = None,
    timeout: Timeout = 120.0,
    retries: Retries = 2,
    retry_backoff: RetryBackoff = 2.0,
    n_samples: NSamples = 1,
    seed: Seed = 42,
    stream: Stream = True,
    output_dir: OutputDir = None,
    memory_gb: MemoryGb = None,
    checkpoint_every: CheckpointEvery = 1,
    progress_interval: ProgressInterval = 5.0,
    resume: Resume = None,
    request_extra_body: ExtraBody = None,
    config: Config = None,
    requests: Annotated[int | None, typer.Option("--requests", min=1)] = None,
    duration: Annotated[float, typer.Option("--duration", min=0)] = 60.0,
    warmup_requests: Annotated[int, typer.Option("--warmup-requests", min=0)] = 0,
    request_rate: RequestRate = None,
    ramp_seconds: RampSeconds = 0.0,
    prompt_profile: PromptProfile = "mixed",
    server_metrics: ServerMetrics = False,
) -> None:
    """Run quality and/or load workloads against one API endpoint."""
    if mode != "load":
        load_options = (
            "requests",
            "duration",
            "warmup_requests",
            "request_rate",
            "ramp_seconds",
            "prompt_profile",
            "server_metrics",
        )
        supplied_load = [
            "--" + name.replace("_", "-")
            for name in load_options
            if (source := ctx.get_parameter_source(name)) and source.name != "DEFAULT"
        ]
        if supplied_load:
            raise typer.BadParameter(f"{', '.join(supplied_load)} require --mode load")
    elif requests is not None and ctx.get_parameter_source("duration").name == "DEFAULT":
        duration = 0.0
    values = {key: value for key, value in locals().items() if key in RunSpec.__slots__}
    values["run_mode"] = mode.value
    if values.get("concurrency") is None:
        values["concurrency"] = 1 if mode == "quality" else 16
    if mode in {"quality", "both"}:
        summary, paths = _run_quality(ctx, values, config_path=config, resume=resume, mode="run")
        _report(summary, paths)
    else:
        if max_tokens is None:
            values["max_tokens"] = 128
        summary, paths = _run_stress_mode(
            ctx,
            {**values, "config": config, "resume": resume},
            concurrency="",
            duration=duration,
            requests=requests,
            warmup_requests=warmup_requests,
            request_rate=request_rate,
            ramp_seconds=ramp_seconds,
            prompt_profile=prompt_profile,
            server_metrics=server_metrics,
        )
        _report(summary, paths)


def _evaluate(ctx: typer.Context, mode: str, arguments: dict[str, Any]) -> None:
    resume = arguments["resume"]
    values = {
        key: value
        for key, value in arguments.items()
        if key not in {"ctx", "config", "resume", "mode", "arguments"}
    }
    summary, paths = _run_quality(
        ctx, values, config_path=arguments["config"], resume=resume, mode=mode
    )
    _report(summary, paths)


@app.command("stress", hidden=True)
def stress_command(
    ctx: typer.Context,
    base_url: BaseUrl = None,
    api_key: ApiKey = None,
    provider: Provider = None,
    api: Api = None,
    model: Model = None,
    concurrency: Annotated[
        str,
        typer.Option("--concurrency", help="One level, or a comma-separated sweep like 1,8,64."),
    ] = "64",
    duration: Annotated[float, typer.Option("--duration", min=0)] = 60.0,
    requests: Annotated[int | None, typer.Option("--requests", min=1)] = None,
    warmup_requests: Annotated[int, typer.Option("--warmup-requests", min=0)] = 0,
    max_tokens: Annotated[int, typer.Option("--max-tokens", min=1)] = 128,
    timeout: Timeout = 120.0,
    retries: Retries = 2,
    retry_backoff: RetryBackoff = 2.0,
    seed: Seed = 42,
    stream: Stream = True,
    output_dir: OutputDir = None,
    checkpoint_every: CheckpointEvery = 1,
    progress_interval: ProgressInterval = 5.0,
    request_rate: RequestRate = None,
    ramp_seconds: RampSeconds = 0.0,
    prompt_profile: PromptProfile = "mixed",
    server_metrics: ServerMetrics = True,
    config: Config = None,
) -> None:
    """Measure throughput and latency without scoring answers.

    Passing several concurrency levels runs them in sequence and adds a sweep report.
    """
    if requests is not None and ctx.get_parameter_source("duration").name == "DEFAULT":
        duration = 0.0
    configured_concurrency = ctx.get_parameter_source("concurrency").name == "DEFAULT"
    summary, paths = _run_stress_mode(
        ctx,
        {
            "base_url": base_url,
            "api_key": api_key,
            "provider": provider,
            "api": api,
            "model": model,
            "concurrency": 64,
            "config": config,
            "max_tokens": max_tokens,
            "timeout": timeout,
            "retries": retries,
            "retry_backoff": retry_backoff,
            "seed": seed,
            "stream": stream,
            "output_dir": output_dir,
            "checkpoint_every": checkpoint_every,
            "progress_interval": progress_interval,
        },
        concurrency="" if configured_concurrency else concurrency,
        duration=duration,
        requests=requests,
        warmup_requests=warmup_requests,
        request_rate=request_rate,
        ramp_seconds=ramp_seconds,
        prompt_profile=prompt_profile,
        server_metrics=server_metrics,
    )
    _report(summary, paths)


@app.command("sweep", hidden=True)
def sweep_command(ctx: typer.Context) -> None:
    """Deprecated: use `llmbench stress --concurrency 1,4,8,16,32,64 --warmup-requests 20`."""
    typer.echo(
        "`llmbench sweep` was merged into `llmbench stress`; pass several concurrency "
        "levels instead, for example:\n"
        "  llmbench stress --concurrency 1,4,8,16,32,64 --warmup-requests 20 --requests 200",
        err=True,
    )
    raise typer.Exit(code=2)


@app.command("suite", hidden=True)
def suite_command(
    config: Annotated[Path, typer.Option("--config", exists=True, readable=True)],
    dataset: Annotated[str | None, typer.Option("--dataset")] = None,
    limit: Annotated[int | None, typer.Option("--limit", min=1)] = None,
    output_dir: OutputDir = None,
) -> None:
    """Run installed representative benchmarks through capability-specific targets."""
    names = tuple(part.strip() for part in dataset.split(",") if part.strip()) if dataset else None
    try:
        summary, paths = run_suite(config, dataset=names, limit=limit, output_dir=output_dir)
    except (SpecError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    _report(summary, paths)


@app.command("compare")
def compare_command(
    baseline: Annotated[Path, typer.Option("--baseline", exists=True, readable=True)],
    candidate: Annotated[Path, typer.Option("--candidate", exists=True, readable=True)],
    report: Annotated[Path, typer.Option("--report")] = Path("reports/compare.html"),
    policy: Annotated[Path | None, typer.Option("--policy", exists=True, readable=True)] = None,
    bootstrap_samples: Annotated[int, typer.Option("--bootstrap-samples", min=100)] = 10_000,
    seed: Seed = 42,
) -> None:
    """Compare paired baseline and candidate run directories."""
    try:
        comparison = compare_run_directories(
            baseline, candidate, bootstrap_samples=bootstrap_samples, seed=seed
        )
        if policy is not None:
            comparison["policy"] = evaluate_policy(comparison, load_yaml(policy))
    except IncomparableRunsError as exc:
        typer.echo(f"Incomparable runs: {exc}", err=True)
        raise typer.Exit(code=3) from exc
    except (OSError, ValueError, KeyError) as exc:
        typer.echo(f"Comparison infrastructure error: {exc}", err=True)
        raise typer.Exit(code=4) from exc
    typer.echo(f"Comparison written to {write_comparison(report, comparison)['html']}")
    if comparison.get("policy") and not comparison["policy"]["passed"]:
        raise typer.Exit(code=2)


@app.command("validate", hidden=True)
def validate_command(
    run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
) -> None:
    """Validate a schema-v2 run directory and raw JSONL records."""
    try:
        validated = validate_run_directory(run_dir)
    except (OSError, ValueError) as exc:
        typer.echo(f"Validation failed: {exc}", err=True)
        raise typer.Exit(code=4) from exc
    typer.echo("Validated: " + ", ".join(validated))


@app.command("datasets")
@app.command("list-datasets", hidden=True)
def list_datasets_command() -> None:
    """List datasets readable right now: bundled plus installed data packs."""
    for item in installed_datasets():
        restriction = f" [{item['restriction']}]" if item.get("restriction") else ""
        image_extra = " [image extra]" if item.get("capability") == "multimodal" else ""
        reports = report_count_for_dataset(item["name"])
        typer.echo(
            f"{item['name']:<14} {item['count']:>5} {item['category']:<24} "
            f"reports={'n/a' if reports is None else reports:<4} "
            f"{item['license']}{restriction}{image_extra}"
        )


@app.command("list-benchmarks", hidden=True)
def list_benchmarks_command(
    category: Annotated[str | None, typer.Option("--category")] = None,
    top: Annotated[int | None, typer.Option("--top", min=1)] = None,
    bundled_only: Annotated[bool, typer.Option("--bundled-only")] = False,
) -> None:
    """List the snapshotted DataLearner catalog, ranked by published reports."""
    rows = list_benchmarks(category=category, bundled_only=bundled_only)
    for item in rows[:top] if top is not None else rows:
        status = f"bundled:{item['bundled_as']}" if item["bundled_as"] else "catalog-only"
        typer.echo(
            f"{item['code']:<34} reports={item['report_count']:<4} {item['category']:<18} {status}"
        )


@app.command("coverage", hidden=True)
def coverage_command() -> None:
    """Show the supported representative benchmark capability matrix."""
    rows = capability_matrix()
    for row in rows:
        state = "installed" if row["installed"] else f"requires:{row['data_pack']}"
        records = "" if row["count"] is None else f" records={row['count']}"
        typer.echo(
            f"{row['category']:<24} {row['benchmark']:<28} "
            f"capability={row['capability']:<12} {state}{records}"
        )
    typer.echo(f"Coverage: {sum(row['installed'] for row in rows)}/{len(rows)} categories")


@app.command("list-models", hidden=True)
def list_models_command(
    base_url: BaseUrl = None,
    api_key: ApiKey = None,
    provider: Provider = None,
    api: Api = None,
) -> None:
    """Discover model IDs from GET /models."""
    try:
        endpoint = resolve_endpoint(base_url=base_url, api_key=api_key, provider=provider, api=api)
        models = discover_models(
            endpoint.base_url, endpoint.api_key, api=endpoint.api, provider=endpoint.provider
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except (httpx.HTTPError, OSError) as exc:
        typer.echo(f"Model discovery failed: {exc}", err=True)
        raise typer.Exit(code=4) from exc
    for model in models:
        typer.echo(model)


@data_app.command("list")
def data_list_command() -> None:
    """List installed optional data wheels."""
    packs = discover_data_packs()
    if not packs:
        typer.echo("No optional data packs installed.")
        return
    for pack in packs:
        typer.echo(
            f"{pack['name']} {pack['version']} datasets={len(pack['datasets'])} "
            f"package={pack['package']}"
        )


@data_app.command("verify")
def data_verify_command() -> None:
    """Verify installed data-wheel hashes and record counts."""
    rows = verify_installed_data_packs()
    if not rows:
        typer.echo("No optional data packs installed.")
        return
    failed = False
    for row in rows:
        valid = row["sha256_valid"] and row["count_valid"]
        failed = failed or not valid
        typer.echo(
            f"{row['dataset']:<30} pack={row['pack']}@{row['version']} "
            f"count={row['count']} status={'ok' if valid else 'FAILED'}"
        )
    if failed:
        raise typer.Exit(code=4)


@executor_app.command("serve")
def executor_serve_command(
    config: Annotated[Path, typer.Option("--config", exists=True, readable=True)],
    host: Annotated[str | None, typer.Option("--host")] = None,
    port: Annotated[int | None, typer.Option("--port", min=1, max=65535)] = None,
) -> None:
    """Serve the remote executor API."""
    from .executor import serve

    try:
        serve(config, host=host, port=port)
    except (RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc


@executor_app.command("smoke")
def executor_smoke_command(
    executor_url: Annotated[str, typer.Option("--executor-url", envvar="EXECUTOR_URL")],
    ephemeral_key: Annotated[str, typer.Option("--ephemeral-key", envvar="EXECUTOR_TASK_KEY")],
    image: Annotated[str, typer.Option("--image")],
    timeout: Timeout = 120.0,
) -> None:
    """Submit a safe smoke task and wait for its artifact."""
    import json

    from .executor_client import smoke

    try:
        artifact = smoke(executor_url, ephemeral_key=ephemeral_key, image=image, timeout=timeout)
    except SmokeFailed as exc:
        typer.echo(json.dumps(exc.job, ensure_ascii=False), err=True)
        raise typer.Exit(code=4) from exc
    typer.echo(json.dumps(artifact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
