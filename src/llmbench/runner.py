from __future__ import annotations

import asyncio
import inspect
import time
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from . import __version__
from .client import OpenAICompatibleClient
from .datasets import load_many, stress_prompts
from .images import prepare_image_messages
from .report import write_sweep_artifacts
from .repro import build_run_manifest, canonical_hash
from .runspec import DEFAULT_MAX_TOKENS, LoadSpec, RunSpec
from .schemas import DatasetItem, RequestResult
from .scoring import build_messages, build_prompt, score_output
from .session import ResumeMismatch, RunSession
from .telemetry import PrometheusCollector, metric_delta


def new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


ResultCallback = Callable[[RequestResult, int, int], Awaitable[None] | None]


class BenchmarkRunner:
    def __init__(
        self,
        *,
        client: OpenAICompatibleClient,
        model: str,
        concurrency: int,
        temperature: float | None,
        top_p: float | None,
        max_tokens: int | None,
        stream: bool,
        seed: int,
        run_id: str | None = None,
        request_extra_body: dict | None = None,
        image_input_hashes: dict[tuple[str, str], str] | None = None,
    ) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        self.client = client
        self.model = model
        self.concurrency = concurrency
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.stream = stream
        self.seed = seed
        self.run_id = run_id or new_run_id()
        self.request_extra_body = dict(request_extra_body or {})
        self.image_input_hashes = image_input_hashes or {}

    async def evaluate(
        self,
        items: list[DatasetItem],
        *,
        n_samples: int = 1,
        existing: list[RequestResult] | None = None,
        on_result: ResultCallback | None = None,
    ) -> tuple[list[RequestResult], float]:
        if n_samples < 1:
            raise ValueError("n_samples must be at least 1")
        started = time.perf_counter()
        results = list(existing or [])
        completed_keys = {result.key for result in results}
        total = len(items) * n_samples
        completed = len(results)
        queue: asyncio.Queue[tuple[DatasetItem, int] | None] = asyncio.Queue(
            maxsize=self.concurrency * 2
        )
        result_lock = asyncio.Lock()

        async def producer() -> None:
            for item in items:
                for sample_id in range(1, n_samples + 1):
                    if (item.dataset, item.id, sample_id) not in completed_keys:
                        await queue.put((item, sample_id))
            for _ in range(self.concurrency):
                await queue.put(None)

        async def worker() -> None:
            nonlocal completed
            while True:
                job = await queue.get()
                try:
                    if job is None:
                        return
                    result = await self._one(*job)
                    async with result_lock:
                        results.append(result)
                        completed += 1
                        current = completed
                    if on_result is not None:
                        callback_result = on_result(result, current, total)
                        if inspect.isawaitable(callback_result):
                            await callback_result
                finally:
                    queue.task_done()

        tasks = [
            asyncio.create_task(producer()),
            *(asyncio.create_task(worker()) for _ in range(self.concurrency)),
        ]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        return results, time.perf_counter() - started

    async def stress(
        self,
        prompts: list[DatasetItem],
        *,
        duration: float,
        max_requests: int | None = None,
        on_result: ResultCallback | None = None,
        request_rate: float | None = None,
        ramp_seconds: float = 0.0,
    ) -> tuple[list[RequestResult], float]:
        if duration <= 0 and max_requests is None:
            raise ValueError("duration must be positive when max_requests is not set")
        started = time.perf_counter()
        deadline = started + duration if duration > 0 else float("inf")
        results: list[RequestResult] = []
        counter = 0
        lock = asyncio.Lock()
        pacing_lock = asyncio.Lock()
        next_start = started

        async def worker(worker_id: int) -> None:
            nonlocal counter, next_start
            if ramp_seconds > 0 and self.concurrency > 1:
                await asyncio.sleep(ramp_seconds * worker_id / (self.concurrency - 1))
            while time.perf_counter() < deadline:
                async with lock:
                    if max_requests is not None and counter >= max_requests:
                        return
                    index = counter
                    counter += 1
                if request_rate is not None:
                    async with pacing_lock:
                        now = time.perf_counter()
                        delay = max(0.0, next_start - now)
                        next_start = max(next_start, now) + 1.0 / request_rate
                    if delay:
                        await asyncio.sleep(delay)
                item = prompts[index % len(prompts)]
                result = await self._one(item, index + 1)
                results.append(result)
                if on_result is not None:
                    callback_result = on_result(
                        result,
                        len(results),
                        max_requests or 0,
                    )
                    if inspect.isawaitable(callback_result):
                        await callback_result

        await asyncio.gather(*(worker(index) for index in range(self.concurrency)))
        results.sort(key=lambda row: row.sample_id)
        return results, time.perf_counter() - started

    async def _one(self, item: DatasetItem, sample_id: int) -> RequestResult:
        capability = str(item.metadata.get("capability") or "chat")
        adapter = str(item.metadata.get("adapter") or "native")
        if capability not in {"chat", "multimodal"} or adapter not in {"native", "multimodal_chat"}:
            return self._unsupported(item, sample_id, f"{capability}/{adapter}")
        images = []
        if item.is_image:
            messages, images = prepare_image_messages(item)
            if not images:
                raise ValueError(f"image question {item.id!r} has no image assets")
            expected = self.image_input_hashes.get((item.dataset, item.id))
            if expected is not None and canonical_hash(messages) != expected:
                raise ValueError(f"image input changed during run: {item.id}")
        else:
            messages = build_messages(item)
        recommended = int(item.metadata.get("recommended_max_tokens") or 4096)
        max_tokens = self.max_tokens if self.max_tokens is not None else max(4096, recommended)
        completion = await self.client.complete(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=max_tokens,
            stream=self.stream,
            seed=self.seed + sample_id - 1,
            extra_body=self.request_extra_body,
        )
        if completion.error:
            parsed, score, parse_failed = None, 0.0 if item.answer is not None else None, False
        else:
            parsed, score, parse_failed = score_output(item, completion.content)
        return RequestResult(
            run_id=self.run_id,
            model=self.model,
            dataset=item.dataset,
            benchmark_category=str(item.metadata.get("benchmark_category") or "Custom"),
            question_type=item.type,
            metric=str(item.metadata.get("benchmark_metric") or "exact_match"),
            question_id=item.id,
            sample_id=sample_id,
            concurrency=self.concurrency,
            prompt=build_prompt(item),
            raw_output=completion.content,
            parsed_answer=parsed,
            gold_answer=item.answer,
            score=score,
            parse_failed=parse_failed,
            latency_ms=completion.latency_ms,
            ttft_ms=completion.ttft_ms,
            tpot_ms=completion.tpot_ms,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            finish_reason=completion.finish_reason,
            error=completion.error,
            error_type=completion.error_type,
            http_status=completion.http_status,
            attempts=completion.attempts,
            max_tokens=max_tokens,
            attempt_latency_ms=completion.attempt_latency_ms,
            reasoning_tokens=completion.reasoning_tokens,
            cached_input_tokens=completion.cached_input_tokens,
            usage_available=completion.usage_available,
            request_id=completion.request_id,
            raw_response=completion.raw_response,
            images=images,
        )

    def _unsupported(self, item: DatasetItem, sample_id: int, capability: str) -> RequestResult:
        recommended = int(item.metadata.get("recommended_max_tokens") or 4096)
        max_tokens = self.max_tokens if self.max_tokens is not None else max(4096, recommended)
        return RequestResult(
            run_id=self.run_id,
            model=self.model,
            dataset=item.dataset,
            benchmark_category=str(item.metadata.get("benchmark_category") or "Custom"),
            question_type=item.type,
            metric=str(item.metadata.get("benchmark_metric") or "none"),
            question_id=item.id,
            sample_id=sample_id,
            concurrency=self.concurrency,
            prompt=build_prompt(item),
            raw_output="",
            parsed_answer=None,
            gold_answer=item.answer,
            score=None,
            parse_failed=False,
            latency_ms=0,
            ttft_ms=None,
            tpot_ms=None,
            input_tokens=0,
            output_tokens=0,
            finish_reason=None,
            error=f"target does not provide required capability: {capability}",
            error_type="unsupported_capability",
            http_status=None,
            attempts=0,
            max_tokens=max_tokens,
            attempt_latency_ms=None,
        )


async def select_model(client: OpenAICompatibleClient, model: str | None) -> tuple[str, list[str]]:
    return await client.resolve_model(model)


def _client(spec: RunSpec) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        base_url=spec.base_url,
        api_key=spec.api_key,
        api=spec.api,
        provider=spec.provider,
        timeout=spec.timeout,
        retries=spec.retries,
        retry_backoff=spec.retry_backoff,
    )


def _session(spec: RunSpec, directory: Path, mode: str) -> RunSession:
    return RunSession(
        directory,
        mode=mode,
        checkpoint_every=spec.checkpoint_every,
        progress_interval=spec.progress_interval,
    )


def run_evaluation(
    spec: RunSpec, *, mode: str, resume: bool = False
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Score a dataset selection and write every run artifact."""
    return asyncio.run(_evaluate(spec, mode=mode, resume=resume))


async def _evaluate(
    spec: RunSpec, *, mode: str, resume: bool
) -> tuple[dict[str, Any], dict[str, Path]]:
    items = load_many(
        list(spec.dataset),
        limit_per_dataset=spec.limit,
        sample=spec.sample,
        seed=spec.seed,
    )
    directory = spec.output_dir or default_output_dir(mode)
    async with _client(spec) as client:
        selected, available = await select_model(client, spec.model)
        session = _session(spec, directory, mode)
        dataset_max_tokens = {
            dataset: spec.resolved_max_tokens(_recommended_max_tokens(items, dataset))
            for dataset in sorted({item.dataset for item in items})
        }
        config = spec.to_config(dataset_max_tokens=dataset_max_tokens, available_models=available)
        run_id = session.writer.load_manifest()["run_id"] if resume else new_run_id()
        manifest = build_run_manifest(
            run_id=run_id,
            mode=mode,
            model=selected,
            base_url=spec.base_url,
            config=config,
            items=items,
            n_samples=spec.n_samples,
        )
        existing = session.open(manifest, resume=resume)
        expected = {
            (item.dataset, item.id, sample_id)
            for item in items
            for sample_id in range(1, spec.n_samples + 1)
        }
        if not {result.key for result in existing} <= expected:
            raise ResumeMismatch("resume results contain requests outside the current manifest")
        for dataset in dataset_max_tokens:
            recommended = _recommended_max_tokens(items, dataset)
            if spec.max_tokens is not None and spec.max_tokens < recommended:
                session.warn(
                    f"warning: {dataset} recommends max_tokens>={recommended}; "
                    f"explicit value is {spec.max_tokens}",
                    "max_tokens_warning",
                    dataset=dataset,
                )
        print(
            f"Running {mode}: model={selected} api={spec.api} datasets={','.join(spec.dataset)} "
            f"questions={len(items)} concurrency={spec.concurrency} resumed={len(existing)}"
        )
        runner = BenchmarkRunner(
            client=client,
            model=selected,
            concurrency=spec.concurrency,
            temperature=spec.temperature,
            top_p=spec.top_p,
            max_tokens=spec.max_tokens,
            stream=spec.stream,
            seed=spec.seed,
            run_id=run_id,
            request_extra_body=spec.request_extra_body,
            image_input_hashes={
                (entry["dataset"], entry["question_id"]): entry["messages_sha256"]
                for entry in manifest.get("image_inputs", [])
            },
        )
        results, elapsed = await runner.evaluate(
            items,
            n_samples=spec.n_samples,
            existing=existing,
            on_result=session.on_result,
        )
        return session.close(
            results,
            elapsed=elapsed,
            run_id=run_id,
            model=selected,
            base_url=spec.base_url,
            config=config,
            total=len(expected),
        )


def run_stress(spec: RunSpec, load: LoadSpec) -> tuple[dict[str, Any], dict[str, Path]]:
    """Measure throughput and latency at one or more concurrency levels."""
    return asyncio.run(_stress(spec, load))


async def _stress(spec: RunSpec, load: LoadSpec) -> tuple[dict[str, Any], dict[str, Path]]:
    prompts = stress_prompts(load.prompt_profile)
    root = spec.output_dir or default_output_dir("stress")
    async with _client(spec) as client:
        selected, available = await select_model(client, spec.model)
        collector = PrometheusCollector(spec.base_url) if load.server_metrics else None
        points = []
        for level in load.levels:
            directory = root / f"c{level}" if load.is_sweep else root
            summary, paths = await _stress_point(
                spec,
                load,
                level=level,
                client=client,
                collector=collector,
                prompts=prompts,
                directory=directory,
                model=selected,
                available_models=available,
            )
            points.append({"concurrency": level, "summary": summary})
        if not load.is_sweep:
            return summary, paths
        sweep_paths = write_sweep_artifacts(
            root,
            {
                "schema_version": 2,
                "model": selected,
                "base_url": spec.base_url,
                "prompt_profile": load.prompt_profile,
                "points": points,
            },
        )
        return summary, {**paths, **{f"sweep_{k}": v for k, v in sweep_paths.items()}}


async def _stress_point(
    spec: RunSpec,
    load: LoadSpec,
    *,
    level: int,
    client: OpenAICompatibleClient,
    collector: PrometheusCollector | None,
    prompts: list[DatasetItem],
    directory: Path,
    model: str,
    available_models: list[str],
) -> tuple[dict[str, Any], dict[str, Path]]:
    runner = BenchmarkRunner(
        client=client,
        model=model,
        concurrency=level,
        temperature=spec.temperature,
        top_p=spec.top_p,
        max_tokens=spec.max_tokens,
        stream=spec.stream,
        seed=spec.seed,
        request_extra_body=spec.request_extra_body,
    )
    if load.warmup_requests:
        await runner.stress(
            prompts,
            duration=0,
            max_requests=load.warmup_requests,
            request_rate=load.request_rate,
            ramp_seconds=load.ramp_seconds,
        )
    config = load.to_config(level, spec, available_models)
    session = _session(spec, directory, "stress")
    session.open(_stress_manifest(runner.run_id, model, spec.base_url, config), resume=False)
    print(
        f"Running stress: model={model} api={spec.api} concurrency={level} "
        f"duration={load.duration}s requests={load.requests}"
    )
    before = await _snapshot(collector, model, session)
    results, elapsed = await runner.stress(
        prompts,
        duration=load.duration,
        max_requests=load.requests,
        on_result=session.on_result,
        request_rate=load.request_rate,
        ramp_seconds=load.ramp_seconds,
    )
    after = await _snapshot(collector, model, session)
    telemetry = None
    if before is not None and after is not None:
        telemetry = {
            "server_telemetry": {
                "url": after["url"],
                "delta": metric_delta(before["metrics"], after["metrics"]),
            }
        }
    return session.close(
        results,
        elapsed=elapsed,
        run_id=runner.run_id,
        model=model,
        base_url=spec.base_url,
        config=config,
        total=load.requests,
        extra_summary=telemetry,
    )


async def _snapshot(
    collector: PrometheusCollector | None, model: str, session: RunSession
) -> dict[str, Any] | None:
    if collector is None:
        return None
    try:
        return await collector.snapshot(model=model)
    except (httpx.HTTPError, ValueError) as exc:
        session.writer.event("telemetry_unavailable", error=str(exc))
        return None


def _stress_manifest(
    run_id: str, model: str, base_url: str, config: dict[str, Any]
) -> dict[str, Any]:
    manifest = {
        "schema_version": 2,
        "run_id": run_id,
        "llmbench_version": __version__,
        "mode": "stress",
        "model": model,
        "base_url": base_url,
        "target_capabilities": ["chat", "stream"] if config.get("stream") else ["chat"],
        "config": config,
    }
    manifest["fingerprint"] = canonical_hash(manifest)
    return manifest


def _recommended_max_tokens(items: list[DatasetItem], dataset: str) -> int:
    return max(
        int(item.metadata.get("recommended_max_tokens") or DEFAULT_MAX_TOKENS)
        for item in items
        if item.dataset == dataset
    )


def default_output_dir(prefix: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("runs") / f"{prefix}-{stamp}"
