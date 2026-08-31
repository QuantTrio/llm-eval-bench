from __future__ import annotations

import asyncio
import inspect
import time
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone

from .client import OpenAICompatibleClient
from .schemas import DatasetItem, RequestResult
from .scoring import build_messages, build_prompt, score_output


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
        temperature: float,
        top_p: float,
        max_tokens: int | None,
        stream: bool,
        seed: int,
        run_id: str | None = None,
        request_extra_body: dict | None = None,
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
        )
