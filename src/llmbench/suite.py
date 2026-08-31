from __future__ import annotations

import time
from typing import Any

from .adapters import (
    humaneval_executor_payload,
    judge_response,
    multimodal_messages,
    score_embedding_item,
)
from .client import OpenAICompatibleClient
from .executor_client import RemoteExecutorClient
from .runner import BenchmarkRunner
from .schemas import DatasetItem, RequestResult
from .scoring import build_messages, build_prompt, score_output


class CapabilityRunner(BenchmarkRunner):
    def __init__(
        self,
        *,
        chat_client: OpenAICompatibleClient,
        chat_model: str,
        multimodal_client: OpenAICompatibleClient | None = None,
        multimodal_model: str | None = None,
        embedding_client: OpenAICompatibleClient | None = None,
        embedding_model: str | None = None,
        judge_client: OpenAICompatibleClient | None = None,
        judge_model: str | None = None,
        judge_repeats: int = 3,
        executor_client: RemoteExecutorClient | None = None,
        executor_key: str | None = None,
        executor_image: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(client=chat_client, model=chat_model, **kwargs)
        self.multimodal_client = multimodal_client
        self.multimodal_model = multimodal_model
        self.embedding_client = embedding_client
        self.embedding_model = embedding_model
        self.judge_client = judge_client
        self.judge_model = judge_model
        self.judge_repeats = judge_repeats
        self.executor_client = executor_client
        self.executor_key = executor_key
        self.executor_image = executor_image

    async def _one(self, item: DatasetItem, sample_id: int) -> RequestResult:
        capability = str(item.metadata.get("capability") or "chat")
        adapter = str(item.metadata.get("adapter") or "native")
        if capability == "chat" and adapter == "native":
            return await super()._one(item, sample_id)
        if adapter == "multimodal_judge":
            return await self._multimodal_judged(item, sample_id)
        if capability == "multimodal" or adapter == "multimodal_chat":
            return await self._multimodal(item, sample_id)
        if capability == "embedding" or adapter == "embedding":
            return await self._embedding(item, sample_id)
        if adapter == "judge":
            return await self._judged(item, sample_id)
        if adapter == "remote_executor" and item.dataset == "humaneval":
            return await self._humaneval(item, sample_id)
        if adapter == "remote_browser":
            return await self._remote_browser(item, sample_id)
        if adapter in {"official_harness", "artifact_judge"}:
            return await self._remote_harness(item, sample_id)
        return self._unsupported(item, sample_id, f"{capability}/{adapter}")

    async def _multimodal(self, item: DatasetItem, sample_id: int) -> RequestResult:
        if self.multimodal_client is None or self.multimodal_model is None:
            return self._unsupported(item, sample_id, "multimodal")
        started = time.perf_counter()
        try:
            messages = multimodal_messages(item)
        except (OSError, ValueError) as exc:
            return self._result(
                item,
                sample_id,
                model=self.multimodal_model,
                error=str(exc),
                error_type="multimodal_asset_error",
            )
        completion = await self.multimodal_client.complete(
            model=self.multimodal_model,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self._max_tokens(item),
            stream=self.stream,
            seed=self.seed + sample_id - 1,
            extra_body=self.request_extra_body,
        )
        parsed, score, parse_failed = score_output(item, completion.content)
        return self._result(
            item,
            sample_id,
            model=self.multimodal_model,
            raw_output=completion.content,
            parsed_answer=parsed,
            score=score,
            parse_failed=parse_failed,
            latency_ms=(time.perf_counter() - started) * 1000,
            ttft_ms=completion.ttft_ms,
            tpot_ms=completion.tpot_ms,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            finish_reason=completion.finish_reason,
            error=completion.error,
            error_type=completion.error_type,
            attempts=completion.attempts,
            attempt_latency_ms=completion.attempt_latency_ms,
        )

    async def _embedding(self, item: DatasetItem, sample_id: int) -> RequestResult:
        if self.embedding_client is None or self.embedding_model is None:
            return self._unsupported(item, sample_id, "embedding")
        started = time.perf_counter()
        try:
            score, details = await score_embedding_item(
                self.embedding_client,
                model=self.embedding_model,
                item=item,
            )
        except ValueError as exc:
            return self._result(
                item,
                sample_id,
                model=self.embedding_model,
                error=str(exc),
                error_type="embedding_item_error",
            )
        return self._result(
            item,
            sample_id,
            model=self.embedding_model,
            raw_output=str(details),
            parsed_answer=None if score is None else str(int(score)),
            score=score,
            latency_ms=(time.perf_counter() - started) * 1000,
            input_tokens=int(details.get("input_tokens") or 0),
            error=details.get("error"),
            error_type=details.get("error_type"),
        )

    async def _multimodal_judged(self, item: DatasetItem, sample_id: int) -> RequestResult:
        if (
            self.multimodal_client is None
            or self.multimodal_model is None
            or self.judge_client is None
            or self.judge_model is None
        ):
            return self._unsupported(item, sample_id, "multimodal/judge")
        started = time.perf_counter()
        try:
            messages = multimodal_messages(item)
        except (OSError, ValueError) as exc:
            return self._result(
                item,
                sample_id,
                model=self.multimodal_model,
                error=str(exc),
                error_type="multimodal_asset_error",
            )
        completion = await self.multimodal_client.complete(
            model=self.multimodal_model,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self._max_tokens(item),
            stream=self.stream,
            seed=self.seed + sample_id - 1,
        )
        if completion.error:
            return self._result(
                item,
                sample_id,
                model=self.multimodal_model,
                error=completion.error,
                error_type=completion.error_type,
                attempts=completion.attempts,
            )
        judged = await judge_response(
            self.judge_client,
            model=self.judge_model,
            item=item,
            candidate_output=completion.content,
            repeats=self.judge_repeats,
        )
        return self._result(
            item,
            sample_id,
            model=self.multimodal_model,
            raw_output=completion.content,
            parsed_answer=None if judged.score is None else str(judged.score),
            score=judged.score,
            parse_failed=judged.score is None,
            latency_ms=(time.perf_counter() - started) * 1000,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            finish_reason=completion.finish_reason,
            error="; ".join(judged.errors) if judged.score is None else None,
            error_type="judge_error" if judged.score is None else None,
            attempts=completion.attempts,
        )

    async def _judged(self, item: DatasetItem, sample_id: int) -> RequestResult:
        if self.judge_client is None or self.judge_model is None:
            return self._unsupported(item, sample_id, "judge")
        started = time.perf_counter()
        completion = await self.client.complete(
            model=self.model,
            messages=build_messages(item),
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self._max_tokens(item),
            stream=self.stream,
            seed=self.seed + sample_id - 1,
            extra_body=self.request_extra_body,
        )
        if completion.error:
            return self._result(
                item,
                sample_id,
                model=self.model,
                latency_ms=(time.perf_counter() - started) * 1000,
                error=completion.error,
                error_type=completion.error_type,
                attempts=completion.attempts,
            )
        judged = await judge_response(
            self.judge_client,
            model=self.judge_model,
            item=item,
            candidate_output=completion.content,
            repeats=self.judge_repeats,
        )
        return self._result(
            item,
            sample_id,
            model=self.model,
            raw_output=completion.content,
            parsed_answer=None if judged.score is None else str(judged.score),
            score=judged.score,
            parse_failed=judged.score is None,
            latency_ms=(time.perf_counter() - started) * 1000,
            ttft_ms=completion.ttft_ms,
            tpot_ms=completion.tpot_ms,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            finish_reason=completion.finish_reason,
            error="; ".join(judged.errors) if judged.score is None else None,
            error_type="judge_error" if judged.score is None else None,
            attempts=completion.attempts,
        )

    async def _humaneval(self, item: DatasetItem, sample_id: int) -> RequestResult:
        if self.executor_client is None or self.executor_key is None or self.executor_image is None:
            return self._unsupported(item, sample_id, "agent/remote_executor")
        started = time.perf_counter()
        completion = await self.client.complete(
            model=self.model,
            messages=build_messages(item),
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self._max_tokens(item),
            stream=False,
            seed=self.seed + sample_id - 1,
            extra_body=self.request_extra_body,
        )
        if completion.error:
            return self._result(
                item,
                sample_id,
                model=self.model,
                error=completion.error,
                error_type=completion.error_type,
                attempts=completion.attempts,
            )
        try:
            payload = humaneval_executor_payload(
                item, completion.content, image=self.executor_image
            )
            submitted = await self.executor_client.submit(payload, ephemeral_key=self.executor_key)
            job = await self.executor_client.wait(submitted["id"])
            artifact = (
                await self.executor_client.artifacts(submitted["id"])
                if job["status"] == "completed"
                else {}
            )
            score = float(job["status"] == "completed" and artifact.get("exit_code") == 0)
            error = None if score else str(job.get("error") or artifact.get("stderr") or "failed")
        except (OSError, ValueError, TimeoutError, KeyError) as exc:
            score = 0.0
            error = str(exc)
            artifact = {}
        return self._result(
            item,
            sample_id,
            model=self.model,
            raw_output=completion.content,
            parsed_answer=str(artifact.get("exit_code")) if artifact else None,
            score=score,
            latency_ms=(time.perf_counter() - started) * 1000,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            finish_reason=completion.finish_reason,
            error=error,
            error_type="executor_error" if error else None,
            attempts=completion.attempts,
        )

    async def _remote_browser(self, item: DatasetItem, sample_id: int) -> RequestResult:
        if self.executor_client is None or self.executor_key is None or self.executor_image is None:
            return self._unsupported(item, sample_id, "agent/remote_browser")
        started = time.perf_counter()
        payload = {
            "image": self.executor_image,
            "command": [
                "-m",
                "llmbench_harness.browsecomp",
                "--problem",
                str(item.metadata.get("encrypted_problem") or ""),
                "--answer",
                str(item.metadata.get("encrypted_answer") or ""),
                "--canary",
                str(item.metadata.get("canary") or ""),
            ],
            "network": True,
        }
        try:
            submitted = await self.executor_client.submit(payload, ephemeral_key=self.executor_key)
            job = await self.executor_client.wait(submitted["id"])
            artifact = (
                await self.executor_client.artifacts(submitted["id"])
                if job["status"] == "completed"
                else {}
            )
            score = float(artifact.get("score")) if "score" in artifact else None
            error = None if score is not None else str(job.get("error") or "missing score")
        except (OSError, ValueError, TimeoutError, KeyError) as exc:
            score = None
            error = str(exc)
            artifact = {}
        return self._result(
            item,
            sample_id,
            model=self.model,
            raw_output="[sensitive browser artifact withheld]",
            parsed_answer=None,
            score=score,
            latency_ms=(time.perf_counter() - started) * 1000,
            error=error,
            error_type="browser_executor_error" if error else None,
            attempts=1,
        )

    async def _remote_harness(self, item: DatasetItem, sample_id: int) -> RequestResult:
        if self.executor_client is None or self.executor_key is None or self.executor_image is None:
            return self._unsupported(item, sample_id, "agent/official_harness")
        command = item.metadata.get("executor_command")
        if not isinstance(command, list) or not command:
            return self._result(
                item,
                sample_id,
                model=self.model,
                error="data pack did not provide executor_command",
                error_type="executor_item_error",
            )
        started = time.perf_counter()
        try:
            submitted = await self.executor_client.submit(
                {
                    "image": str(item.metadata.get("executor_image") or self.executor_image),
                    "command": [str(value) for value in command],
                    "network": bool(item.metadata.get("network", False)),
                },
                ephemeral_key=self.executor_key,
            )
            job = await self.executor_client.wait(submitted["id"])
            artifact = (
                await self.executor_client.artifacts(submitted["id"])
                if job["status"] == "completed"
                else {}
            )
            score = float(artifact["score"]) if "score" in artifact else None
            error = None if score is not None else str(job.get("error") or "missing score")
        except (OSError, ValueError, TimeoutError, KeyError) as exc:
            score = None
            error = str(exc)
            artifact = {}
        return self._result(
            item,
            sample_id,
            model=self.model,
            raw_output=str(artifact.get("summary") or ""),
            parsed_answer=None,
            score=score,
            latency_ms=(time.perf_counter() - started) * 1000,
            error=error,
            error_type="executor_error" if error else None,
        )

    def _max_tokens(self, item: DatasetItem) -> int:
        recommended = int(item.metadata.get("recommended_max_tokens") or 4096)
        return self.max_tokens if self.max_tokens is not None else max(4096, recommended)

    def _result(
        self,
        item: DatasetItem,
        sample_id: int,
        *,
        model: str,
        raw_output: str = "",
        parsed_answer: str | None = None,
        score: float | None = None,
        parse_failed: bool = False,
        latency_ms: float = 0,
        ttft_ms: float | None = None,
        tpot_ms: float | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        finish_reason: str | None = None,
        error: str | None = None,
        error_type: str | None = None,
        attempts: int = 1,
        attempt_latency_ms: float | None = None,
    ) -> RequestResult:
        return RequestResult(
            run_id=self.run_id,
            model=model,
            dataset=item.dataset,
            benchmark_category=str(item.metadata.get("benchmark_category") or "Custom"),
            question_type=item.type,
            metric=str(item.metadata.get("benchmark_metric") or "none"),
            question_id=item.id,
            sample_id=sample_id,
            concurrency=self.concurrency,
            prompt=build_prompt(item),
            raw_output=raw_output,
            parsed_answer=parsed_answer,
            gold_answer=item.answer,
            score=score,
            parse_failed=parse_failed,
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason=finish_reason,
            error=error,
            error_type=error_type,
            http_status=None,
            attempts=attempts,
            max_tokens=self._max_tokens(item),
            attempt_latency_ms=attempt_latency_ms,
        )
