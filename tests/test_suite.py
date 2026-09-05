from __future__ import annotations

import json

import httpx
import pytest

from llmbench.client import OpenAICompatibleClient
from llmbench.schemas import DatasetItem
from llmbench.suite import CapabilityRunner


class FakeExecutor:
    async def submit(self, payload, *, ephemeral_key: str):
        assert ephemeral_key == "temporary"
        if payload["command"][0] == "-c":
            assert "check(answer)" in payload["command"][1]
        elif "browsecomp" in payload["command"]:
            assert "llmbench_harness.browsecomp" in payload["command"]
        return {"id": "job"}

    async def wait(self, job_id: str):
        return {"id": job_id, "status": "completed"}

    async def artifacts(self, job_id: str):
        return {"exit_code": 0, "score": 1, "stdout": "ok", "job_id": job_id}


@pytest.mark.asyncio
async def test_capability_runner_routes_all_core_adapters(tmp_path) -> None:
    from PIL import Image

    image = tmp_path / "image.png"
    Image.new("RGB", (2, 2), "white").save(image)
    document = tmp_path / "document.pdf"
    document.write_bytes(b"pdf")

    async def candidate_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.path.endswith("/embeddings"):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"index": 0, "embedding": [1, 0]},
                        {"index": 1, "embedding": [1, 0]},
                        {"index": 2, "embedding": [0, 1]},
                    ]
                },
            )
        content = body["messages"][0]["content"]
        if isinstance(content, list):
            answer = '{"answer":"A"}'
        elif "def answer" in content:
            answer = "    return 42"
        else:
            answer = "candidate response"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": answer}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3},
            },
        )

    async def judge_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"score":1}'}, "finish_reason": "stop"}]},
        )

    async with (
        OpenAICompatibleClient(
            base_url="http://chat/v1",
            api_key="EMPTY",
            transport=httpx.MockTransport(candidate_handler),
        ) as chat,
        OpenAICompatibleClient(
            base_url="http://multimodal/v1",
            api_key="EMPTY",
            transport=httpx.MockTransport(candidate_handler),
        ) as multimodal,
        OpenAICompatibleClient(
            base_url="http://embedding/v1",
            api_key="EMPTY",
            transport=httpx.MockTransport(candidate_handler),
        ) as embedding,
        OpenAICompatibleClient(
            base_url="http://judge/v1",
            api_key="EMPTY",
            transport=httpx.MockTransport(judge_handler),
        ) as judge,
    ):
        runner = CapabilityRunner(
            chat_client=chat,
            chat_model="chat-model",
            multimodal_client=multimodal,
            multimodal_model="vision-model",
            embedding_client=embedding,
            embedding_model="embed-model",
            judge_client=judge,
            judge_model="judge-model",
            judge_repeats=3,
            executor_client=FakeExecutor(),  # type: ignore[arg-type]
            executor_key="temporary",
            executor_image="sandbox:test",
            concurrency=1,
            temperature=0,
            top_p=1,
            max_tokens=4096,
            stream=False,
            seed=42,
        )
        items = [
            DatasetItem(
                id="mm",
                dataset="mmmu",
                type="multiple_choice",
                question="What is shown?",
                choices={"A": "answer", "B": "other"},
                answer="A",
                metadata={
                    "capability": "multimodal",
                    "adapter": "multimodal_chat",
                    "assets": [str(image)],
                    "benchmark_metric": "accuracy",
                },
            ),
            DatasetItem(
                id="embed",
                dataset="mteb",
                type="embedding",
                question="query",
                metadata={
                    "capability": "embedding",
                    "adapter": "embedding",
                    "positive": "right",
                    "negatives": ["wrong"],
                    "benchmark_metric": "recall_at_1",
                },
            ),
            DatasetItem(
                id="hle",
                dataset="hle",
                type="judge",
                question="Answer the multimodal question",
                answer="reference",
                metadata={
                    "capability": "multimodal",
                    "adapter": "multimodal_judge",
                    "assets": [str(image)],
                    "rubric": "correctness",
                    "benchmark_metric": "judge_accuracy",
                },
            ),
            DatasetItem(
                id="bad-mm",
                dataset="mmmu",
                type="multiple_choice",
                question="What is shown?",
                metadata={
                    "capability": "multimodal",
                    "adapter": "multimodal_chat",
                    "assets": [str(document)],
                },
            ),
            DatasetItem(
                id="bad-embed",
                dataset="mteb",
                type="embedding",
                question="query",
                metadata={"capability": "embedding", "adapter": "embedding"},
            ),
            DatasetItem(
                id="judge",
                dataset="simpleqa",
                type="judge",
                question="Question",
                answer="reference",
                metadata={
                    "capability": "chat",
                    "adapter": "judge",
                    "rubric": "correctness",
                    "benchmark_metric": "judge_accuracy",
                },
            ),
            DatasetItem(
                id="HumanEval/0",
                dataset="humaneval",
                type="code",
                question="def answer():\n",
                answer="    return 42",
                metadata={
                    "capability": "agent",
                    "adapter": "remote_executor",
                    "entry_point": "answer",
                    "test": "def check(fn): assert fn() == 42",
                    "benchmark_metric": "pass_at_1",
                },
            ),
            DatasetItem(
                id="browse",
                dataset="browsecomp",
                type="agent",
                question="[encrypted]",
                metadata={
                    "capability": "agent",
                    "adapter": "remote_browser",
                    "encrypted_problem": "cipher-problem",
                    "encrypted_answer": "cipher-answer",
                    "canary": "canary",
                    "benchmark_metric": "judge_accuracy",
                },
            ),
            DatasetItem(
                id="harness",
                dataset="terminal-bench-2",
                type="agent",
                question="task",
                metadata={
                    "capability": "agent",
                    "adapter": "official_harness",
                    "executor_command": ["run-task", "task"],
                    "benchmark_metric": "success_rate",
                },
            ),
            DatasetItem(
                id="bad-harness",
                dataset="terminal-bench-2",
                type="agent",
                question="task",
                metadata={
                    "capability": "agent",
                    "adapter": "official_harness",
                    "benchmark_metric": "success_rate",
                },
            ),
        ]
        results, _ = await runner.evaluate(items)
    assert [result.score for result in results] == [
        1.0,
        1.0,
        1.0,
        None,
        None,
        1.0,
        1.0,
        1.0,
        1.0,
        None,
    ]
    assert results[3].error_type == "multimodal_asset_error"
    assert results[4].error_type == "embedding_item_error"


@pytest.mark.asyncio
async def test_capability_runner_marks_missing_targets_unsupported() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}]},
        )

    async with OpenAICompatibleClient(
        base_url="http://chat/v1",
        api_key="EMPTY",
        transport=httpx.MockTransport(handler),
    ) as chat:
        runner = CapabilityRunner(
            chat_client=chat,
            chat_model="chat-model",
            concurrency=1,
            temperature=0,
            top_p=1,
            max_tokens=4096,
            stream=False,
            seed=42,
        )
        items = [
            DatasetItem(
                id="mm",
                dataset="mmmu",
                type="multiple_choice",
                question="question",
                metadata={"capability": "multimodal", "adapter": "multimodal_chat"},
            ),
            DatasetItem(
                id="embedding",
                dataset="mteb",
                type="embedding",
                question="query",
                metadata={"capability": "embedding", "adapter": "embedding"},
            ),
            DatasetItem(
                id="hle",
                dataset="hle",
                type="judge",
                question="question",
                metadata={"capability": "multimodal", "adapter": "multimodal_judge"},
            ),
            DatasetItem(
                id="judge",
                dataset="simpleqa",
                type="judge",
                question="question",
                metadata={"capability": "chat", "adapter": "judge"},
            ),
            DatasetItem(
                id="code",
                dataset="humaneval",
                type="code",
                question="code",
                metadata={"capability": "agent", "adapter": "remote_executor"},
            ),
            DatasetItem(
                id="unknown",
                dataset="unknown",
                type="unknown",
                question="unknown",
                metadata={"capability": "other", "adapter": "other"},
            ),
            DatasetItem(
                id="browse",
                dataset="browsecomp",
                type="agent",
                question="encrypted",
                metadata={"capability": "agent", "adapter": "remote_browser"},
            ),
        ]
        results, _ = await runner.evaluate(items)
    assert all(result.error_type == "unsupported_capability" for result in results)
