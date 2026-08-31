from __future__ import annotations

import asyncio
import contextlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import httpx
import typer

from . import __version__
from .artifacts import RunArtifactWriter, utc_now
from .capabilities import capability_matrix
from .catalog import list_benchmarks, report_count_for_dataset
from .client import OpenAICompatibleClient
from .comparison import IncomparableRunsError, compare_run_directories, evaluate_policy
from .config import load_bench_config, load_yaml, secret_from_env
from .data_packs import discover_data_packs, verify_installed_data_packs
from .datasets import list_datasets as dataset_catalog
from .datasets import load_many, stress_prompts
from .executor_client import RemoteExecutorClient
from .metrics import summarize
from .report import write_comparison, write_run_artifacts, write_sweep_artifacts
from .repro import build_run_manifest, canonical_hash
from .runner import BenchmarkRunner
from .suite import CapabilityRunner
from .telemetry import PrometheusCollector, metric_delta
from .validation import validate_run_directory

app = typer.Typer(
    name="llmbench",
    help="Quality and concurrency benchmarks for OpenAI-compatible LLM APIs.",
    no_args_is_help=True,
)
executor_app = typer.Typer(help="Serve and manage the remote isolated task executor.")
app.add_typer(executor_app, name="executor")
data_app = typer.Typer(help="Inspect and verify optional benchmark data wheels.")
app.add_typer(data_app, name="data")

BaseUrl = Annotated[
    str | None,
    typer.Option("--base-url", envvar="OPENAI_BASE_URL", help="API root ending in /v1."),
]
ApiKey = Annotated[
    str | None,
    typer.Option("--api-key", envvar="OPENAI_API_KEY", help="API key; use EMPTY if allowed."),
]


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


def _required(value: str | None, option: str) -> str:
    if value:
        return value
    raise typer.BadParameter(f"{option} is required (or set its OPENAI_* environment variable)")


def _dataset_names(value: str) -> list[str]:
    names = [part.strip() for part in value.split(",") if part.strip()]
    if not names:
        raise typer.BadParameter("at least one dataset is required")
    return names


def _default_output(prefix: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("runs") / f"{prefix}-{stamp}"


def _print_run_summary(summary: dict, paths: dict[str, Path]) -> None:
    quality = summary["quality"]
    performance = summary["performance"]
    score = quality.get("composite_score")
    if score is None:
        score = quality.get("sample_mean_score", quality.get("mean_score"))
    score_text = "n/a" if score is None else f"{score:.4f}"
    typer.echo(f"Model: {summary['model']}")
    typer.echo(
        f"Requests: {performance['total_requests']}  "
        f"Successful: {performance['successful_requests']}  Score: {score_text}"
    )
    typer.echo(f"Summary: {paths['summary']}")
    typer.echo(f"Raw results: {paths['raw']}")
    typer.echo(f"Markdown: {paths['markdown']}")
    typer.echo(f"HTML: {paths['html']}")


def _parse_extra_body(value: str | None) -> dict:
    if value is None:
        return {}
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"--request-extra-body must be valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise typer.BadParameter("--request-extra-body must be a JSON object")
    protected = {"model", "messages", "stream"} & set(payload)
    if protected:
        raise typer.BadParameter(
            "--request-extra-body cannot override: " + ", ".join(sorted(protected))
        )
    return payload


def _merge_bench_config(
    ctx: typer.Context,
    config_path: Path | None,
    values: dict,
) -> dict:
    if config_path is None or values.get("resume") is not None:
        return values
    try:
        config = load_bench_config(config_path)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    target = (config.get("targets") or {}).get("chat") or {}
    run = config.get("run") or {}
    configured: dict[str, object] = {}
    if target.get("base_url") is not None:
        configured["base_url"] = target["base_url"]
    if target.get("model") is not None:
        configured["model"] = target["model"]
    if target.get("api_key_env") is not None:
        configured["api_key"] = secret_from_env(str(target["api_key_env"]))
    run_mapping = {
        "datasets": "dataset",
        "limit_per_dataset": "limit",
        "limit": "limit",
        "sample": "sample",
        "concurrency": "concurrency",
        "temperature": "temperature",
        "top_p": "top_p",
        "max_tokens": "max_tokens",
        "timeout": "timeout",
        "retries": "retries",
        "retry_backoff": "retry_backoff",
        "n_samples": "n_samples",
        "seed": "seed",
        "stream": "stream",
        "output_dir": "output_dir",
        "memory_gb": "memory_gb",
        "checkpoint_every": "checkpoint_every",
        "progress_interval": "progress_interval",
        "request_extra_body": "request_extra_body",
    }
    for source, target_name in run_mapping.items():
        if source in run:
            configured[target_name] = run[source]
    if isinstance(configured.get("dataset"), list):
        configured["dataset"] = ",".join(str(item) for item in configured["dataset"])
    if isinstance(configured.get("output_dir"), str):
        configured["output_dir"] = Path(str(configured["output_dir"]))
    if isinstance(configured.get("request_extra_body"), dict):
        configured["request_extra_body"] = json.dumps(configured["request_extra_body"])
    for name, value in configured.items():
        source = ctx.get_parameter_source(name)
        if source is None or source.name == "DEFAULT":
            values[name] = value
    return values


async def _resolve(client: OpenAICompatibleClient, model: str | None) -> tuple[str, list[str]]:
    selected, available = await client.resolve_model(model)
    if model is None and len(available) > 1:
        typer.echo(
            f"Multiple models discovered; using first model '{selected}'. Use --model to override.",
            err=True,
        )
    return selected, available


async def _run_eval_mode(
    *,
    mode: str,
    base_url: str,
    api_key: str,
    model: str | None,
    datasets: list[str],
    limit: int,
    sample: int | None,
    concurrency: int,
    temperature: float,
    top_p: float,
    max_tokens: int | None,
    timeout: float,
    retries: int,
    retry_backoff: float,
    n_samples: int,
    seed: int,
    stream: bool,
    output_dir: Path,
    memory_gb: float | None,
    checkpoint_every: int,
    progress_interval: float,
    resume: bool,
    request_extra_body: dict,
) -> None:
    items = load_many(datasets, limit_per_dataset=limit, sample=sample, seed=seed)
    async with OpenAICompatibleClient(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        retries=retries,
        retry_backoff=retry_backoff,
    ) as client:
        selected, available = await _resolve(client, model)
        writer = RunArtifactWriter(output_dir, checkpoint_every=checkpoint_every)
        previous_manifest = writer.load_manifest() if resume else None
        previous_state = writer.load_state() if resume else {}
        previous_elapsed = float(previous_state.get("elapsed_seconds") or 0.0)
        runner = BenchmarkRunner(
            client=client,
            model=selected,
            concurrency=concurrency,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=stream,
            seed=seed,
            run_id=previous_manifest["run_id"] if previous_manifest else None,
            request_extra_body=request_extra_body,
        )
        dataset_max_tokens = {
            dataset: (
                max_tokens
                if max_tokens is not None
                else max(
                    4096,
                    max(
                        int(item.metadata.get("recommended_max_tokens") or 4096)
                        for item in items
                        if item.dataset == dataset
                    ),
                )
            )
            for dataset in sorted({item.dataset for item in items})
        }
        config = {
            "datasets": datasets,
            "limit_per_dataset": limit,
            "sample": sample,
            "concurrency": concurrency,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "default_max_tokens": 4096,
            "dataset_max_tokens": dataset_max_tokens,
            "timeout": timeout,
            "retries": retries,
            "retry_backoff": retry_backoff,
            "n_samples": n_samples,
            "seed": seed,
            "stream": stream,
            "memory_gb": memory_gb,
            "available_models": available,
            "request_extra_body": request_extra_body,
            "checkpoint_every": checkpoint_every,
            "progress_interval": progress_interval,
        }
        manifest = build_run_manifest(
            run_id=runner.run_id,
            mode=mode,
            model=selected,
            base_url=base_url,
            config=config,
            items=items,
            n_samples=n_samples,
        )
        if previous_manifest and previous_manifest.get("fingerprint") != manifest["fingerprint"]:
            raise typer.BadParameter(
                "resume configuration does not match run_manifest.json; "
                "start a new output directory"
            )
        if not previous_manifest:
            writer.write_manifest(manifest)
        existing = writer.existing_results() if resume else []
        completed_keys = {result.key for result in existing}
        if len(completed_keys) != len(existing):
            raise typer.BadParameter("resume results contain duplicate request keys")
        expected_keys = {
            (item.dataset, item.id, sample_id)
            for item in items
            for sample_id in range(1, n_samples + 1)
        }
        if not completed_keys <= expected_keys:
            raise typer.BadParameter("resume results contain requests outside the current manifest")

        for dataset in dataset_max_tokens:
            recommended = max(
                int(item.metadata.get("recommended_max_tokens") or 4096)
                for item in items
                if item.dataset == dataset
            )
            if max_tokens is not None and max_tokens < recommended:
                message = (
                    f"warning: {dataset} recommends max_tokens>={recommended}; "
                    f"explicit value is {max_tokens}"
                )
                typer.echo(message, err=True)
                writer.event("max_tokens_warning", dataset=dataset, message=message)

        typer.echo(
            f"Running {mode}: model={selected} datasets={','.join(datasets)} "
            f"questions={len(items)} concurrency={concurrency} "
            f"resumed={len(existing)}"
        )
        writer.event(
            "run_resumed" if resume else "run_started",
            run_id=runner.run_id,
            completed=len(existing),
            total=len(expected_keys),
        )
        session_started = time.perf_counter()
        last_progress = 0.0
        counters = {
            "errors": sum(result.error is not None for result in existing),
            "truncated": sum(result.finish_reason == "length" for result in existing),
            "output_tokens": sum(result.output_tokens for result in existing),
        }

        def on_result(result, completed: int, total: int) -> None:
            nonlocal last_progress
            counters["errors"] += result.error is not None
            counters["truncated"] += result.finish_reason == "length"
            counters["output_tokens"] += result.output_tokens
            elapsed_now = previous_elapsed + time.perf_counter() - session_started
            writer.append_result(
                result,
                completed=completed,
                total=total,
                elapsed_seconds=elapsed_now,
            )
            writer.event(
                "request_completed",
                dataset=result.dataset,
                question_id=result.question_id,
                sample_id=result.sample_id,
                completed=completed,
                total=total,
                error_type=result.error_type,
                truncated=result.finish_reason == "length",
                attempts=result.attempts,
            )
            now = time.perf_counter()
            if now - last_progress < progress_interval and completed != total:
                return
            last_progress = now
            session_elapsed = max(now - session_started, 1e-9)
            session_completed = max(completed - len(existing), 0)
            qps = session_completed / session_elapsed
            eta = (total - completed) / qps if qps > 0 else None
            payload = {
                "event": "progress",
                "completed": completed,
                "total": total,
                "qps": round(qps, 3),
                "eta_seconds": None if eta is None else round(eta, 1),
                **counters,
            }
            if sys.stderr.isatty():
                typer.echo(
                    "\r"
                    f"[{completed}/{total}] qps={qps:.2f} eta={payload['eta_seconds']}s "
                    f"errors={counters['errors']} truncated={counters['truncated']}",
                    nl=completed == total,
                    err=True,
                )
            else:
                typer.echo(json.dumps(payload, ensure_ascii=False), err=True)
            writer.event(
                "progress", **{key: value for key, value in payload.items() if key != "event"}
            )

        results, session_elapsed = await runner.evaluate(
            items,
            n_samples=n_samples,
            existing=existing,
            on_result=on_result,
        )
        elapsed = previous_elapsed + session_elapsed
        summary = summarize(
            results,
            run_id=runner.run_id,
            mode=mode,
            model=selected,
            base_url=base_url,
            elapsed_seconds=elapsed,
            config=config,
        )
        paths = write_run_artifacts(output_dir, summary, results)
        writer.write_state(
            {
                "status": "completed",
                "completed": len(results),
                "total": len(expected_keys),
                "elapsed_seconds": elapsed,
                "updated_at": utc_now(),
            }
        )
        writer.event("run_completed", run_id=runner.run_id, completed=len(results))
        _print_run_summary(summary, paths)


@app.command("list-datasets")
def list_datasets_command() -> None:
    """List datasets bundled in the installed wheel."""
    for item in dataset_catalog():
        restriction = f" [{item['restriction']}]" if item.get("restriction") else ""
        reports = report_count_for_dataset(item["name"])
        report_text = "n/a" if reports is None else str(reports)
        typer.echo(
            f"{item['name']:<14} {item['count']:>5} {item['category']:<24} "
            f"reports={report_text:<4} {item['license']}{restriction}"
        )


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


@app.command("list-benchmarks")
def list_benchmarks_command(
    category: Annotated[str | None, typer.Option("--category")] = None,
    top: Annotated[int | None, typer.Option("--top", min=1)] = None,
    bundled_only: Annotated[bool, typer.Option("--bundled-only")] = False,
) -> None:
    """List the snapshotted DataLearner catalog, ranked by published reports."""
    rows = list_benchmarks(category=category, bundled_only=bundled_only)
    if top is not None:
        rows = rows[:top]
    for item in rows:
        status = f"bundled:{item['bundled_as']}" if item["bundled_as"] else "catalog-only"
        typer.echo(
            f"{item['code']:<34} reports={item['report_count']:<4} {item['category']:<18} {status}"
        )


@app.command("coverage")
def coverage_command() -> None:
    """Show the 23-category representative benchmark capability matrix."""
    installed = {item["name"] for item in dataset_catalog()}
    rows = capability_matrix(installed)
    for row in rows:
        state = "installed" if row["installed"] else f"requires:{row['data_pack']}"
        typer.echo(
            f"{row['category']:<24} {row['benchmark']:<28} "
            f"capability={row['capability']:<12} {state}"
        )
    typer.echo(f"Coverage: {sum(row['installed'] for row in rows)}/{len(rows)} categories")


@app.command("validate")
def validate_command(
    run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
) -> None:
    """Validate a schema-v2 run directory and raw JSONL records."""
    try:
        validated = validate_run_directory(run_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        typer.echo(f"Validation failed: {exc}", err=True)
        raise typer.Exit(code=4) from exc
    typer.echo("Validated: " + ", ".join(validated))


@app.command("list-models")
def list_models_command(base_url: BaseUrl = None, api_key: ApiKey = None) -> None:
    """Discover model IDs from GET /models."""

    async def run() -> None:
        async with OpenAICompatibleClient(
            base_url=_required(base_url, "--base-url"),
            api_key=_required(api_key, "--api-key"),
        ) as client:
            for model in await client.list_models():
                typer.echo(model)

    asyncio.run(run())


@executor_app.command("serve")
def executor_serve_command(
    config: Annotated[Path, typer.Option("--config", exists=True, readable=True)],
    host: Annotated[str | None, typer.Option("--host")] = None,
    port: Annotated[int | None, typer.Option("--port", min=1, max=65535)] = None,
) -> None:
    """Serve the remote executor API."""
    try:
        import uvicorn

        from .executor import ExecutorConfig, create_executor_app
    except ImportError as exc:
        raise typer.BadParameter("install quanttrio-llmbench[executor]") from exc
    payload = load_yaml(config)
    values = payload.get("executor") or payload
    images = values.get("allowed_images") or []
    if not isinstance(images, list) or not images:
        raise typer.BadParameter("executor config requires allowed_images")
    executor_config = ExecutorConfig(
        allowed_images={str(image) for image in images},
        work_dir=Path(values.get("work_dir") or ".llmbench-executor"),
        allow_insecure=bool(values.get("allow_insecure", False)),
        allow_network=bool(values.get("allow_network", False)),
        network_name=values.get("network_name"),
        network_proxy=values.get("network_proxy"),
        allowed_domains=tuple(str(item) for item in values.get("allowed_domains", [])),
        docker_bin=str(values.get("docker_bin") or "docker"),
        cpus=float(values.get("cpus", 1.0)),
        memory=str(values.get("memory") or "2g"),
        pids_limit=int(values.get("pids_limit", 128)),
        timeout_seconds=float(values.get("timeout_seconds", 300)),
        output_limit_bytes=int(values.get("output_limit_bytes", 1_000_000)),
    )
    uvicorn.run(
        create_executor_app(executor_config),
        host=host or str(values.get("host") or "127.0.0.1"),
        port=port or int(values.get("port") or 8765),
        proxy_headers=True,
    )


@executor_app.command("smoke")
def executor_smoke_command(
    executor_url: Annotated[str, typer.Option("--executor-url", envvar="EXECUTOR_URL")],
    ephemeral_key: Annotated[str, typer.Option("--ephemeral-key", envvar="EXECUTOR_TASK_KEY")],
    image: Annotated[str, typer.Option("--image")],
    timeout: Annotated[float, typer.Option("--timeout", min=0.1)] = 120.0,
) -> None:
    """Submit a safe smoke task and wait for its artifact."""
    from .executor_client import RemoteExecutorClient

    async def run() -> None:
        async with RemoteExecutorClient(executor_url, timeout=timeout) as client:
            job = await client.submit(
                {
                    "image": image,
                    "command": ["-c", "print('llmbench executor ok')"],
                    "network": False,
                },
                ephemeral_key=ephemeral_key,
            )
            completed = await client.wait(job["id"], timeout=timeout)
            if completed["status"] != "completed":
                typer.echo(json.dumps(completed, ensure_ascii=False), err=True)
                raise typer.Exit(code=4)
            artifact = await client.artifacts(job["id"])
            typer.echo(json.dumps(artifact, ensure_ascii=False, indent=2))

    asyncio.run(run())


def _evaluation_command(
    mode: str,
    base_url: str | None,
    api_key: str | None,
    model: str | None,
    dataset: str,
    limit: int,
    sample: int | None,
    concurrency: int,
    temperature: float,
    top_p: float,
    max_tokens: int | None,
    timeout: float,
    retries: int,
    retry_backoff: float,
    n_samples: int,
    seed: int,
    stream: bool,
    output_dir: Path | None,
    memory_gb: float | None,
    checkpoint_every: int,
    progress_interval: float,
    resume: Path | None,
    request_extra_body: str | None,
) -> None:
    extra_body = _parse_extra_body(request_extra_body)
    if resume is not None:
        manifest_path = resume / "run_manifest.json"
        if not manifest_path.exists():
            raise typer.BadParameter(f"resume manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("mode") != mode:
            raise typer.BadParameter(
                f"cannot resume mode {manifest.get('mode')!r} with command {mode!r}"
            )
        stored = manifest["config"]
        base_url = str(manifest["base_url"])
        model = str(manifest["model"])
        dataset = ",".join(stored["datasets"])
        limit = int(stored["limit_per_dataset"])
        sample = stored.get("sample")
        concurrency = int(stored["concurrency"])
        temperature = float(stored["temperature"])
        top_p = float(stored["top_p"])
        max_tokens = stored.get("max_tokens")
        timeout = float(stored["timeout"])
        retries = int(stored["retries"])
        retry_backoff = float(stored.get("retry_backoff", retry_backoff))
        n_samples = int(stored["n_samples"])
        seed = int(stored["seed"])
        stream = bool(stored["stream"])
        memory_gb = stored.get("memory_gb")
        extra_body = dict(stored.get("request_extra_body") or {})
        checkpoint_every = int(stored.get("checkpoint_every", checkpoint_every))
        progress_interval = float(stored.get("progress_interval", progress_interval))
        output_dir = resume
    asyncio.run(
        _run_eval_mode(
            mode=mode,
            base_url=_required(base_url, "--base-url"),
            api_key=_required(api_key, "--api-key"),
            model=model,
            datasets=_dataset_names(dataset),
            limit=limit,
            sample=sample,
            concurrency=concurrency,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            timeout=timeout,
            retries=retries,
            retry_backoff=retry_backoff,
            n_samples=n_samples,
            seed=seed,
            stream=stream,
            output_dir=output_dir or _default_output(mode),
            memory_gb=memory_gb,
            checkpoint_every=checkpoint_every,
            progress_interval=progress_interval,
            resume=resume is not None,
            request_extra_body=extra_body,
        )
    )


@app.command("eval")
def eval_command(
    ctx: typer.Context,
    base_url: BaseUrl = None,
    api_key: ApiKey = None,
    model: Annotated[str | None, typer.Option("--model")] = None,
    dataset: Annotated[str, typer.Option("--dataset")] = (
        "mmlu-pro,mmlu-redux,gpqa-diamond,gsm8k,ceval,hellaswag,truthfulqa,drop"
    ),
    limit: Annotated[int, typer.Option("--limit", min=1)] = 100,
    sample: Annotated[int | None, typer.Option("--sample", min=1)] = None,
    concurrency: Annotated[int, typer.Option("--concurrency", min=1)] = 1,
    temperature: Annotated[float, typer.Option("--temperature", min=0)] = 0.0,
    top_p: Annotated[float, typer.Option("--top-p", min=0, max=1)] = 1.0,
    max_tokens: Annotated[
        int | None,
        typer.Option(
            "--max-tokens", min=1, help="Defaults to 4096 or a higher dataset recommendation."
        ),
    ] = None,
    timeout: Annotated[float, typer.Option("--timeout", min=0.1)] = 120.0,
    retries: Annotated[int, typer.Option("--retries", min=0)] = 2,
    retry_backoff: Annotated[float, typer.Option("--retry-backoff", min=0)] = 2.0,
    n_samples: Annotated[int, typer.Option("--n-samples", min=1)] = 1,
    seed: Annotated[int, typer.Option("--seed")] = 42,
    stream: Annotated[bool, typer.Option("--stream/--no-stream")] = False,
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
    memory_gb: Annotated[float | None, typer.Option("--memory-gb", min=0)] = None,
    checkpoint_every: Annotated[int, typer.Option("--checkpoint-every", min=1)] = 1,
    progress_interval: Annotated[float, typer.Option("--progress-interval", min=0.1)] = 5.0,
    resume: Annotated[Path | None, typer.Option("--resume", exists=True, file_okay=False)] = None,
    request_extra_body: Annotated[str | None, typer.Option("--request-extra-body")] = None,
    config: Annotated[Path | None, typer.Option("--config", exists=True, readable=True)] = None,
) -> None:
    """Run a low-concurrency quality evaluation."""
    values = _merge_bench_config(
        ctx,
        config,
        {
            "base_url": base_url,
            "api_key": api_key,
            "model": model,
            "dataset": dataset,
            "limit": limit,
            "sample": sample,
            "concurrency": concurrency,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "timeout": timeout,
            "retries": retries,
            "retry_backoff": retry_backoff,
            "n_samples": n_samples,
            "seed": seed,
            "stream": stream,
            "output_dir": output_dir,
            "memory_gb": memory_gb,
            "checkpoint_every": checkpoint_every,
            "progress_interval": progress_interval,
            "resume": resume,
            "request_extra_body": request_extra_body,
        },
    )
    _evaluation_command("eval", **values)


@app.command("run")
def run_command(
    ctx: typer.Context,
    base_url: BaseUrl = None,
    api_key: ApiKey = None,
    model: Annotated[str | None, typer.Option("--model")] = None,
    dataset: Annotated[str, typer.Option("--dataset")] = (
        "mmlu-pro,mmlu-redux,gpqa-diamond,gsm8k,ceval,hellaswag,truthfulqa,drop"
    ),
    limit: Annotated[int, typer.Option("--limit", min=1)] = 100,
    sample: Annotated[int | None, typer.Option("--sample", min=1)] = None,
    concurrency: Annotated[int, typer.Option("--concurrency", min=1)] = 16,
    temperature: Annotated[float, typer.Option("--temperature", min=0)] = 0.0,
    top_p: Annotated[float, typer.Option("--top-p", min=0, max=1)] = 1.0,
    max_tokens: Annotated[
        int | None,
        typer.Option(
            "--max-tokens", min=1, help="Defaults to 4096 or a higher dataset recommendation."
        ),
    ] = None,
    timeout: Annotated[float, typer.Option("--timeout", min=0.1)] = 120.0,
    retries: Annotated[int, typer.Option("--retries", min=0)] = 2,
    retry_backoff: Annotated[float, typer.Option("--retry-backoff", min=0)] = 2.0,
    n_samples: Annotated[int, typer.Option("--n-samples", min=1)] = 1,
    seed: Annotated[int, typer.Option("--seed")] = 42,
    stream: Annotated[bool, typer.Option("--stream/--no-stream")] = True,
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
    memory_gb: Annotated[float | None, typer.Option("--memory-gb", min=0)] = None,
    checkpoint_every: Annotated[int, typer.Option("--checkpoint-every", min=1)] = 1,
    progress_interval: Annotated[float, typer.Option("--progress-interval", min=0.1)] = 5.0,
    resume: Annotated[Path | None, typer.Option("--resume", exists=True, file_okay=False)] = None,
    request_extra_body: Annotated[str | None, typer.Option("--request-extra-body")] = None,
    config: Annotated[Path | None, typer.Option("--config", exists=True, readable=True)] = None,
) -> None:
    """Evaluate answer quality under concurrent load."""
    values = _merge_bench_config(
        ctx,
        config,
        {
            "base_url": base_url,
            "api_key": api_key,
            "model": model,
            "dataset": dataset,
            "limit": limit,
            "sample": sample,
            "concurrency": concurrency,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "timeout": timeout,
            "retries": retries,
            "retry_backoff": retry_backoff,
            "n_samples": n_samples,
            "seed": seed,
            "stream": stream,
            "output_dir": output_dir,
            "memory_gb": memory_gb,
            "checkpoint_every": checkpoint_every,
            "progress_interval": progress_interval,
            "resume": resume,
            "request_extra_body": request_extra_body,
        },
    )
    _evaluation_command("run", **values)


@app.command("suite")
def suite_command(
    config: Annotated[Path, typer.Option("--config", exists=True, readable=True)],
    dataset: Annotated[str | None, typer.Option("--dataset")] = None,
    limit: Annotated[int | None, typer.Option("--limit", min=1)] = None,
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
) -> None:
    """Run installed representative benchmarks through capability-specific targets."""
    try:
        configuration = load_bench_config(config)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    targets = configuration.get("targets") or {}
    run_config = configuration.get("run") or {}
    chat = targets.get("chat") or {}
    if not chat.get("base_url"):
        raise typer.BadParameter("suite requires targets.chat.base_url")
    available = {item["name"] for item in dataset_catalog()}
    if dataset is None:
        selected_datasets = [
            row["dataset_id"] for row in capability_matrix(available) if row["installed"]
        ]
    else:
        selected_datasets = _dataset_names(dataset)
    if not selected_datasets:
        raise typer.BadParameter("suite has no installed representative datasets")
    selected_limit = limit or int(run_config.get("limit_per_dataset") or 100)
    items = load_many(
        selected_datasets,
        limit_per_dataset=selected_limit,
        sample=None,
        seed=int(run_config.get("seed", 42)),
    )
    directory = output_dir or Path(run_config.get("output_dir") or _default_output("suite"))

    async def run() -> None:
        async with contextlib.AsyncExitStack() as stack:

            async def target_client(name: str):
                values = targets.get(name) or {}
                if not values.get("base_url"):
                    return None, None
                key = secret_from_env(values.get("api_key_env")) or "EMPTY"
                client = await stack.enter_async_context(
                    OpenAICompatibleClient(
                        base_url=str(values["base_url"]),
                        api_key=key,
                        timeout=float(values.get("timeout", run_config.get("timeout", 300))),
                        retries=int(values.get("retries", run_config.get("retries", 2))),
                    )
                )
                model = values.get("model")
                if model is None:
                    model, _ = await client.resolve_model(None)
                return client, str(model)

            chat_client, chat_model = await target_client("chat")
            if chat_client is None or chat_model is None:
                raise typer.BadParameter("suite requires a usable chat target")
            multimodal_client, multimodal_model = await target_client("multimodal")
            embedding_client, embedding_model = await target_client("embedding")
            judge_client, judge_model = await target_client("judge")
            judge_config = configuration.get("judge") or {}
            if judge_client is None and judge_config.get("base_url"):
                judge_key = secret_from_env(judge_config.get("api_key_env")) or "EMPTY"
                judge_client = await stack.enter_async_context(
                    OpenAICompatibleClient(
                        base_url=str(judge_config["base_url"]),
                        api_key=judge_key,
                        timeout=float(judge_config.get("timeout", 300)),
                        retries=int(judge_config.get("retries", 2)),
                    )
                )
                judge_model = str(judge_config["model"])
            agent = targets.get("agent") or {}
            executor_client = None
            executor_key = None
            if agent.get("executor_url"):
                executor_client = await stack.enter_async_context(
                    RemoteExecutorClient(
                        str(agent["executor_url"]),
                        timeout=float(agent.get("timeout", 600)),
                    )
                )
                executor_key = secret_from_env(agent.get("ephemeral_key_env"))

            runner = CapabilityRunner(
                chat_client=chat_client,
                chat_model=chat_model,
                multimodal_client=multimodal_client,
                multimodal_model=multimodal_model,
                embedding_client=embedding_client,
                embedding_model=embedding_model,
                judge_client=judge_client,
                judge_model=judge_model,
                judge_repeats=int(judge_config.get("repeats", 3)),
                executor_client=executor_client,
                executor_key=executor_key,
                executor_image=agent.get("image"),
                concurrency=int(run_config.get("concurrency", 16)),
                temperature=float(run_config.get("temperature", 0)),
                top_p=float(run_config.get("top_p", 1)),
                max_tokens=run_config.get("max_tokens"),
                stream=bool(run_config.get("stream", True)),
                seed=int(run_config.get("seed", 42)),
                request_extra_body=dict(run_config.get("request_extra_body") or {}),
            )
            writer = RunArtifactWriter(
                directory,
                checkpoint_every=int(run_config.get("checkpoint_every", 1)),
            )
            manifest = build_run_manifest(
                run_id=runner.run_id,
                mode="suite",
                model=chat_model,
                base_url=str(chat["base_url"]),
                config={
                    "datasets": selected_datasets,
                    "limit_per_dataset": selected_limit,
                    "sample": None,
                    "concurrency": runner.concurrency,
                    "temperature": runner.temperature,
                    "top_p": runner.top_p,
                    "max_tokens": runner.max_tokens,
                    "timeout": run_config.get("timeout", 300),
                    "retries": run_config.get("retries", 2),
                    "retry_backoff": run_config.get("retry_backoff", 2),
                    "n_samples": 1,
                    "seed": runner.seed,
                    "stream": runner.stream,
                    "request_extra_body": run_config.get("request_extra_body") or {},
                    "target_capabilities": sorted(targets),
                },
                items=items,
                n_samples=1,
            )
            manifest["target_capabilities"] = sorted(targets)
            writer.write_manifest(manifest)
            writer.event(
                "run_started",
                run_id=runner.run_id,
                completed=0,
                total=len(items),
            )
            started = time.perf_counter()

            def persist(result, completed: int, total: int) -> None:
                writer.append_result(
                    result,
                    completed=completed,
                    total=total,
                    elapsed_seconds=time.perf_counter() - started,
                )
                if completed == total or completed % 10 == 0:
                    typer.echo(
                        json.dumps(
                            {
                                "event": "suite_progress",
                                "completed": completed,
                                "total": total,
                                "unsupported": result.error_type == "unsupported_capability",
                            }
                        ),
                        err=True,
                    )

            results, elapsed = await runner.evaluate(items, on_result=persist)
            summary_config = {
                "datasets": selected_datasets,
                "limit_per_dataset": selected_limit,
                "concurrency": runner.concurrency,
                "temperature": runner.temperature,
                "top_p": runner.top_p,
                "max_tokens": runner.max_tokens,
                "n_samples": 1,
                "seed": runner.seed,
                "stream": runner.stream,
                "target_capabilities": sorted(targets),
                "judge_repeats": runner.judge_repeats,
            }
            summary = summarize(
                results,
                run_id=runner.run_id,
                mode="suite",
                model=chat_model,
                base_url=str(chat["base_url"]),
                elapsed_seconds=elapsed,
                config=summary_config,
            )
            requested_categories = sorted({result.benchmark_category for result in results})
            supported_categories = sorted(
                {
                    result.benchmark_category
                    for result in results
                    if result.error_type != "unsupported_capability"
                }
            )
            summary["coverage"] = {
                "requested_categories": requested_categories,
                "supported_categories": supported_categories,
                "unsupported_categories": sorted(
                    set(requested_categories) - set(supported_categories)
                ),
                "ratio": (
                    len(supported_categories) / len(requested_categories)
                    if requested_categories
                    else 0
                ),
            }
            paths = write_run_artifacts(directory, summary, results)
            writer.write_state(
                {
                    "status": "completed",
                    "completed": len(results),
                    "total": len(results),
                    "elapsed_seconds": elapsed,
                    "updated_at": utc_now(),
                }
            )
            writer.event("run_completed", run_id=runner.run_id, completed=len(results))
            _print_run_summary(summary, paths)

    asyncio.run(run())


@app.command("stress")
def stress_command(
    base_url: BaseUrl = None,
    api_key: ApiKey = None,
    model: Annotated[str | None, typer.Option("--model")] = None,
    concurrency: Annotated[int, typer.Option("--concurrency", min=1)] = 64,
    duration: Annotated[float, typer.Option("--duration", min=0)] = 60.0,
    requests: Annotated[int | None, typer.Option("--requests", min=1)] = None,
    max_tokens: Annotated[int, typer.Option("--max-tokens", min=1)] = 128,
    timeout: Annotated[float, typer.Option("--timeout", min=0.1)] = 120.0,
    retries: Annotated[int, typer.Option("--retries", min=0)] = 2,
    retry_backoff: Annotated[float, typer.Option("--retry-backoff", min=0)] = 2.0,
    seed: Annotated[int, typer.Option("--seed")] = 42,
    stream: Annotated[bool, typer.Option("--stream/--no-stream")] = True,
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
    checkpoint_every: Annotated[int, typer.Option("--checkpoint-every", min=1)] = 1,
    progress_interval: Annotated[float, typer.Option("--progress-interval", min=0.1)] = 5.0,
    request_rate: Annotated[float | None, typer.Option("--request-rate", min=0.01)] = None,
    ramp_seconds: Annotated[float, typer.Option("--ramp-seconds", min=0)] = 0.0,
    prompt_profile: Annotated[str, typer.Option("--prompt-profile")] = "mixed",
    server_metrics: Annotated[bool, typer.Option("--server-metrics/--no-server-metrics")] = True,
) -> None:
    """Measure throughput and latency without scoring answers."""

    async def run() -> None:
        url = _required(base_url, "--base-url")
        try:
            prompts = stress_prompts(prompt_profile)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        async with OpenAICompatibleClient(
            base_url=url,
            api_key=_required(api_key, "--api-key"),
            timeout=timeout,
            retries=retries,
            retry_backoff=retry_backoff,
        ) as client:
            selected, available = await _resolve(client, model)
            runner = BenchmarkRunner(
                client=client,
                model=selected,
                concurrency=concurrency,
                temperature=0,
                top_p=1,
                max_tokens=max_tokens,
                stream=stream,
                seed=seed,
            )
            typer.echo(
                f"Running stress: model={selected} concurrency={concurrency} duration={duration}s"
            )
            config = {
                "datasets": ["stress"],
                "concurrency": concurrency,
                "duration": duration,
                "max_requests": requests,
                "max_tokens": max_tokens,
                "timeout": timeout,
                "retries": retries,
                "retry_backoff": retry_backoff,
                "seed": seed,
                "stream": stream,
                "available_models": available,
                "checkpoint_every": checkpoint_every,
                "progress_interval": progress_interval,
                "request_rate": request_rate,
                "ramp_seconds": ramp_seconds,
                "prompt_profile": prompt_profile,
                "server_metrics": server_metrics,
            }
            directory = output_dir or _default_output("stress")
            writer = RunArtifactWriter(directory, checkpoint_every=checkpoint_every)
            manifest_payload = {
                "schema_version": 2,
                "run_id": runner.run_id,
                "llmbench_version": __version__,
                "mode": "stress",
                "model": selected,
                "base_url": url,
                "target_capabilities": ["chat", "stream"] if stream else ["chat"],
                "config": config,
            }
            manifest_payload["fingerprint"] = canonical_hash(manifest_payload)
            writer.write_manifest(manifest_payload)
            writer.event(
                "run_started",
                run_id=runner.run_id,
                completed=0,
                total=requests,
            )
            started = time.perf_counter()
            last_progress = 0.0
            collector = PrometheusCollector(url) if server_metrics else None
            telemetry_before = None
            if collector is not None:
                try:
                    telemetry_before = await collector.snapshot(model=selected)
                except (httpx.HTTPError, ValueError) as exc:
                    writer.event("telemetry_unavailable", error=str(exc))

            def on_result(result, completed: int, total: int) -> None:
                nonlocal last_progress
                elapsed_now = time.perf_counter() - started
                writer.append_result(
                    result,
                    completed=completed,
                    total=total,
                    elapsed_seconds=elapsed_now,
                )
                writer.event(
                    "request_completed",
                    question_id=result.question_id,
                    sample_id=result.sample_id,
                    completed=completed,
                    total=total or None,
                    error_type=result.error_type,
                    truncated=result.finish_reason == "length",
                )
                now = time.perf_counter()
                if now - last_progress < progress_interval and (not total or completed != total):
                    return
                last_progress = now
                qps = completed / max(elapsed_now, 1e-9)
                payload = {
                    "event": "progress",
                    "completed": completed,
                    "total": total or None,
                    "qps": round(qps, 3),
                }
                typer.echo(json.dumps(payload), err=True)
                writer.event("progress", completed=completed, total=total or None, qps=qps)

            results, elapsed = await runner.stress(
                prompts,
                duration=duration,
                max_requests=requests,
                on_result=on_result,
                request_rate=request_rate,
                ramp_seconds=ramp_seconds,
            )
            summary = summarize(
                results,
                run_id=runner.run_id,
                mode="stress",
                model=selected,
                base_url=url,
                elapsed_seconds=elapsed,
                config=config,
            )
            if collector is not None and telemetry_before is not None:
                try:
                    telemetry_after = await collector.snapshot(model=selected)
                    summary["server_telemetry"] = {
                        "url": telemetry_after["url"],
                        "delta": metric_delta(
                            telemetry_before["metrics"], telemetry_after["metrics"]
                        ),
                    }
                except (httpx.HTTPError, ValueError) as exc:
                    writer.event("telemetry_unavailable", error=str(exc))
            paths = write_run_artifacts(directory, summary, results)
            writer.write_state(
                {
                    "status": "completed",
                    "completed": len(results),
                    "total": requests,
                    "elapsed_seconds": elapsed,
                    "updated_at": utc_now(),
                }
            )
            writer.event("run_completed", run_id=runner.run_id, completed=len(results))
            _print_run_summary(summary, paths)

    asyncio.run(run())


@app.command("sweep")
def sweep_command(
    base_url: BaseUrl = None,
    api_key: ApiKey = None,
    model: Annotated[str | None, typer.Option("--model")] = None,
    concurrency: Annotated[str, typer.Option("--concurrency")] = "1,4,8,16,32,64",
    warmup_requests: Annotated[int, typer.Option("--warmup-requests", min=0)] = 20,
    requests: Annotated[int, typer.Option("--requests", min=1)] = 200,
    request_rate: Annotated[float | None, typer.Option("--request-rate", min=0.01)] = None,
    ramp_seconds: Annotated[float, typer.Option("--ramp-seconds", min=0)] = 0.0,
    prompt_profile: Annotated[str, typer.Option("--prompt-profile")] = "mixed",
    max_tokens: Annotated[int, typer.Option("--max-tokens", min=1)] = 128,
    timeout: Annotated[float, typer.Option("--timeout", min=0.1)] = 120.0,
    retries: Annotated[int, typer.Option("--retries", min=0)] = 2,
    seed: Annotated[int, typer.Option("--seed")] = 42,
    output_dir: Annotated[Path, typer.Option("--output-dir")] = Path("runs/sweep"),
    server_metrics: Annotated[bool, typer.Option("--server-metrics/--no-server-metrics")] = True,
) -> None:
    """Run warmup followed by a concurrency sweep."""
    try:
        levels = [int(value.strip()) for value in concurrency.split(",") if value.strip()]
    except ValueError as exc:
        raise typer.BadParameter("--concurrency must be comma-separated integers") from exc
    if not levels or any(level < 1 for level in levels):
        raise typer.BadParameter("--concurrency values must all be at least 1")
    try:
        prompts = stress_prompts(prompt_profile)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    async def run() -> None:
        url = _required(base_url, "--base-url")
        key = _required(api_key, "--api-key")
        async with OpenAICompatibleClient(
            base_url=url,
            api_key=key,
            timeout=timeout,
            retries=retries,
        ) as client:
            selected, _ = await _resolve(client, model)
            points = []
            collector = PrometheusCollector(url) if server_metrics else None
            for level in levels:
                runner = BenchmarkRunner(
                    client=client,
                    model=selected,
                    concurrency=level,
                    temperature=0,
                    top_p=1,
                    max_tokens=max_tokens,
                    stream=True,
                    seed=seed,
                )
                if warmup_requests:
                    await runner.stress(
                        prompts,
                        duration=0,
                        max_requests=warmup_requests,
                        request_rate=request_rate,
                        ramp_seconds=ramp_seconds,
                    )
                before = None
                if collector is not None:
                    try:
                        before = await collector.snapshot(model=selected)
                    except (httpx.HTTPError, ValueError):
                        before = None
                point_dir = output_dir / f"c{level}"
                writer = RunArtifactWriter(point_dir)
                point_started = time.perf_counter()
                writer.event(
                    "run_started",
                    run_id=runner.run_id,
                    completed=0,
                    total=requests,
                )

                def persist(
                    result,
                    completed: int,
                    total: int,
                    point_writer=writer,
                    started_at=point_started,
                ) -> None:
                    point_writer.append_result(
                        result,
                        completed=completed,
                        total=total,
                        elapsed_seconds=time.perf_counter() - started_at,
                    )

                typer.echo(
                    f"Sweep concurrency={level}: warmup={warmup_requests} requests={requests}"
                )
                results, elapsed = await runner.stress(
                    prompts,
                    duration=0,
                    max_requests=requests,
                    on_result=persist,
                    request_rate=request_rate,
                    ramp_seconds=ramp_seconds,
                )
                summary = summarize(
                    results,
                    run_id=runner.run_id,
                    mode="sweep",
                    model=selected,
                    base_url=url,
                    elapsed_seconds=elapsed,
                    config={
                        "datasets": ["stress"],
                        "concurrency": level,
                        "warmup_requests": warmup_requests,
                        "requests": requests,
                        "request_rate": request_rate,
                        "ramp_seconds": ramp_seconds,
                        "prompt_profile": prompt_profile,
                        "max_tokens": max_tokens,
                        "stream": True,
                    },
                )
                if collector is not None and before is not None:
                    try:
                        after = await collector.snapshot(model=selected)
                        summary["server_telemetry"] = {
                            "url": after["url"],
                            "delta": metric_delta(before["metrics"], after["metrics"]),
                        }
                    except (httpx.HTTPError, ValueError):
                        pass
                write_run_artifacts(point_dir, summary, results)
                writer.write_state(
                    {
                        "status": "completed",
                        "completed": len(results),
                        "total": requests,
                        "elapsed_seconds": elapsed,
                        "updated_at": utc_now(),
                    }
                )
                writer.event("run_completed", run_id=runner.run_id, completed=len(results))
                points.append({"concurrency": level, "summary": summary})
            payload = {
                "schema_version": 2,
                "model": selected,
                "base_url": url,
                "prompt_profile": prompt_profile,
                "points": points,
            }
            paths = write_sweep_artifacts(output_dir, payload)
            typer.echo(f"Sweep report: {paths['html']}")

    asyncio.run(run())


@app.command("compare")
def compare_command(
    baseline: Annotated[Path, typer.Option("--baseline", exists=True, readable=True)],
    candidate: Annotated[Path, typer.Option("--candidate", exists=True, readable=True)],
    report: Annotated[Path, typer.Option("--report")] = Path("reports/compare.html"),
    policy: Annotated[Path | None, typer.Option("--policy", exists=True, readable=True)] = None,
    bootstrap_samples: Annotated[int, typer.Option("--bootstrap-samples", min=100)] = 10_000,
    seed: Annotated[int, typer.Option("--seed")] = 42,
) -> None:
    """Compare paired baseline and candidate run directories."""
    try:
        comparison = compare_run_directories(
            baseline,
            candidate,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        if policy is not None:
            policy_result = evaluate_policy(comparison, load_yaml(policy))
            comparison["policy"] = policy_result
    except IncomparableRunsError as exc:
        typer.echo(f"Incomparable runs: {exc}", err=True)
        raise typer.Exit(code=3) from exc
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        typer.echo(f"Comparison infrastructure error: {exc}", err=True)
        raise typer.Exit(code=4) from exc
    paths = write_comparison(report, comparison)
    typer.echo(f"Comparison written to {paths['html']}")
    if comparison.get("policy") and not comparison["policy"]["passed"]:
        raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
