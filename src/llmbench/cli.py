from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import typer

from . import __version__
from .catalog import list_benchmarks, report_count_for_dataset
from .client import OpenAICompatibleClient
from .datasets import list_datasets as dataset_catalog
from .datasets import load_many
from .metrics import summarize
from .report import compare_summaries, write_comparison, write_run_artifacts
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
    score = quality.get("mean_score")
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
    max_tokens: int,
    timeout: float,
    retries: int,
    retry_backoff: float,
    n_samples: int,
    seed: int,
    stream: bool,
    output_dir: Path,
    memory_gb: float | None,
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
        runner = BenchmarkRunner(
            client=client,
            model=selected,
            concurrency=concurrency,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=stream,
            seed=seed,
        )
        typer.echo(
            f"Running {mode}: model={selected} datasets={','.join(datasets)} "
            f"questions={len(items)} concurrency={concurrency}"
        )
        results, elapsed = await runner.evaluate(items, n_samples=n_samples)
        config = {
            "datasets": datasets,
            "limit_per_dataset": limit,
            "sample": sample,
            "concurrency": concurrency,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "timeout": timeout,
            "retries": retries,
            "n_samples": n_samples,
            "seed": seed,
            "stream": stream,
            "memory_gb": memory_gb,
            "available_models": available,
        }
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
    max_tokens: int,
    timeout: float,
    retries: int,
    retry_backoff: float,
    n_samples: int,
    seed: int,
    stream: bool,
    output_dir: Path | None,
    memory_gb: float | None,
) -> None:
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
    max_tokens: Annotated[int, typer.Option("--max-tokens", min=1)] = 1024,
    timeout: Annotated[float, typer.Option("--timeout", min=0.1)] = 120.0,
    retries: Annotated[int, typer.Option("--retries", min=0)] = 2,
    retry_backoff: Annotated[float, typer.Option("--retry-backoff", min=0)] = 2.0,
    n_samples: Annotated[int, typer.Option("--n-samples", min=1)] = 1,
    seed: Annotated[int, typer.Option("--seed")] = 42,
    stream: Annotated[bool, typer.Option("--stream/--no-stream")] = False,
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
    memory_gb: Annotated[float | None, typer.Option("--memory-gb", min=0)] = None,
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
    max_tokens: Annotated[int, typer.Option("--max-tokens", min=1)] = 1024,
    timeout: Annotated[float, typer.Option("--timeout", min=0.1)] = 120.0,
    retries: Annotated[int, typer.Option("--retries", min=0)] = 2,
    retry_backoff: Annotated[float, typer.Option("--retry-backoff", min=0)] = 2.0,
    n_samples: Annotated[int, typer.Option("--n-samples", min=1)] = 1,
    seed: Annotated[int, typer.Option("--seed")] = 42,
    stream: Annotated[bool, typer.Option("--stream/--no-stream")] = True,
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
    memory_gb: Annotated[float | None, typer.Option("--memory-gb", min=0)] = None,
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
            results, elapsed = await runner.stress(
                prompts, duration=duration, max_requests=requests
            )
            config = {
                "datasets": ["stress"],
                "concurrency": concurrency,
                "duration": duration,
                "max_requests": requests,
                "max_tokens": max_tokens,
                "timeout": timeout,
                "retries": retries,
                "seed": seed,
                "stream": stream,
                "available_models": available,
            }
            summary = summarize(
                results,
                run_id=runner.run_id,
                mode="stress",
                model=selected,
                base_url=url,
                elapsed_seconds=elapsed,
                config=config,
            )
            paths = write_run_artifacts(output_dir or _default_output("stress"), summary, results)
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
