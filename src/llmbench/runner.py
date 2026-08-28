from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone

from .client import OpenAICompatibleClient
from .schemas import DatasetItem, RequestResult
from .scoring import build_messages, build_prompt, score_output


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


class BenchmarkRunner:
    def __init__(
        self,
        *,
        client: OpenAICompatibleClient,
        model: str,
        concurrency: int,
        temperature: float,
        top_p: float,
        max_tokens: int,
        stream: bool,
        seed: int,
        run_id: str | None = None,
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

    async def evaluate(
        self, items: list[DatasetItem], *, n_samples: int = 1
    ) -> tuple[list[RequestResult], float]:
        if n_samples < 1:
            raise ValueError("n_samples must be at least 1")
        semaphore = asyncio.Semaphore(self.concurrency)
        started = time.perf_counter()

        async def bounded(item: DatasetItem, sample_id: int) -> RequestResult:
            async with semaphore:
                return await self._one(item, sample_id)

        tasks = [
            asyncio.create_task(bounded(item, sample_id))
            for item in items
            for sample_id in range(1, n_samples + 1)
        ]
        results = await asyncio.gather(*tasks)
        return results, time.perf_counter() - started

    async def stress(
        self,
        prompts: list[DatasetItem],
        *,
        duration: float,
        max_requests: int | None = None,
    ) -> tuple[list[RequestResult], float]:
        if duration <= 0 and max_requests is None:
            raise ValueError("duration must be positive when max_requests is not set")
        started = time.perf_counter()
        deadline = started + duration if duration > 0 else float("inf")
        results: list[RequestResult] = []
        counter = 0
        lock = asyncio.Lock()

        async def worker() -> None:
            nonlocal counter
            while time.perf_counter() < deadline:
                async with lock:
                    if max_requests is not None and counter >= max_requests:
                        return
                    index = counter
                    counter += 1
                item = prompts[index % len(prompts)]
                results.append(await self._one(item, index + 1))

        await asyncio.gather(*(worker() for _ in range(self.concurrency)))
        results.sort(key=lambda row: row.sample_id)
        return results, time.perf_counter() - started

    async def _one(self, item: DatasetItem, sample_id: int) -> RequestResult:
        messages = build_messages(item)
        completion = await self.client.complete(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stream=self.stream,
            seed=self.seed + sample_id - 1,
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
        )
