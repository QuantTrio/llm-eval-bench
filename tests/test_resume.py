from __future__ import annotations

import pytest

from llmbench.artifacts import RunArtifactWriter
from llmbench.runner import BenchmarkRunner
from llmbench.schemas import CompletionResult, DatasetItem


class FakeClient:
    async def complete(self, **_: object) -> CompletionResult:
        return CompletionResult(
            content="Paris",
            latency_ms=1,
            input_tokens=3,
            output_tokens=1,
            finish_reason="stop",
        )


def make_runner() -> BenchmarkRunner:
    return BenchmarkRunner(
        client=FakeClient(),  # type: ignore[arg-type]
        model="fake-model",
        concurrency=1,
        temperature=0,
        top_p=1,
        max_tokens=None,
        stream=False,
        seed=42,
        run_id="resume-test",
    )


@pytest.mark.asyncio
async def test_incremental_checkpoint_and_strict_resume(tmp_path) -> None:
    items = [
        DatasetItem(
            id=f"q{index}",
            dataset="custom",
            type="exact_match",
            question="Capital of France?",
            answer="Paris",
        )
        for index in range(5)
    ]
    writer = RunArtifactWriter(tmp_path, checkpoint_every=1)

    def interrupt_after_three(result, completed: int, total: int) -> None:
        writer.append_result(
            result,
            completed=completed,
            total=total,
            elapsed_seconds=float(completed),
        )
        if completed == 3:
            raise InterruptedError("simulated process interruption")

    with pytest.raises(InterruptedError):
        await make_runner().evaluate(items, on_result=interrupt_after_three)

    existing = writer.existing_results()
    assert len(existing) == 3

    def persist_remaining(result, completed: int, total: int) -> None:
        writer.append_result(
            result,
            completed=completed,
            total=total,
            elapsed_seconds=float(completed),
        )

    results, _ = await make_runner().evaluate(
        items,
        existing=existing,
        on_result=persist_remaining,
    )
    assert len(results) == 5
    assert len({result.key for result in results}) == 5
    assert len(writer.existing_results()) == 5


@pytest.mark.asyncio
async def test_stress_request_rate_and_ramp() -> None:
    prompts = [
        DatasetItem(
            id="stress",
            dataset="stress",
            type="stress",
            question="test",
        )
    ]
    runner = make_runner()
    runner.concurrency = 2
    results, _ = await runner.stress(
        prompts,
        duration=0,
        max_requests=4,
        request_rate=1000,
        ramp_seconds=0.001,
    )
    assert len(results) == 4
