from __future__ import annotations

import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import typer

from . import __version__
from .artifacts import RunArtifactWriter, utc_now
from .catalog import list_benchmarks, report_count_for_dataset
from .client import OpenAICompatibleClient
from .datasets import list_datasets as dataset_catalog
from .datasets import load_many
from .metrics import summarize
from .report import compare_summaries, write_comparison, write_run_artifacts
from .repro import build_run_manifest, canonical_hash
from .runner import BenchmarkRunner

app = typer.Typer(
    name="llmbench",
    help="Quality and concurrency benchmarks for OpenAI-compatible LLM APIs.",
    no_args_is_help=True,
)

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
) -> None:
    """Run a low-concurrency quality evaluation."""
    _evaluation_command(
        "eval",
        base_url,
        api_key,
        model,
        dataset,
        limit,
        sample,
        concurrency,
        temperature,
        top_p,
        max_tokens,
        timeout,
        retries,
        retry_backoff,
        n_samples,
        seed,
        stream,
        output_dir,
        memory_gb,
        checkpoint_every,
        progress_interval,
        resume,
        request_extra_body,
    )


@app.command("run")
def run_command(
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
) -> None:
    """Evaluate answer quality under concurrent load."""
    _evaluation_command(
        "run",
        base_url,
        api_key,
        model,
        dataset,
        limit,
        sample,
        concurrency,
        temperature,
        top_p,
        max_tokens,
        timeout,
        retries,
        retry_backoff,
        n_samples,
        seed,
        stream,
        output_dir,
        memory_gb,
        checkpoint_every,
        progress_interval,
        resume,
        request_extra_body,
    )


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
) -> None:
    """Measure throughput and latency without scoring answers."""

    async def run() -> None:
        url = _required(base_url, "--base-url")
        prompts = load_many(["stress"], limit_per_dataset=None, sample=None, seed=seed)
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


@app.command("compare")
def compare_command(
    baseline: Annotated[Path, typer.Option("--baseline", exists=True, readable=True)],
    candidate: Annotated[Path, typer.Option("--candidate", exists=True, readable=True)],
    report: Annotated[Path, typer.Option("--report")] = Path("reports/compare.html"),
) -> None:
    """Compare baseline and candidate summary.json files."""
    base_summary = json.loads(baseline.read_text(encoding="utf-8"))
    candidate_summary = json.loads(candidate.read_text(encoding="utf-8"))
    comparison = compare_summaries(base_summary, candidate_summary)
    paths = write_comparison(report, comparison)
    typer.echo(f"Comparison written to {paths['html']}")


if __name__ == "__main__":
    app()
