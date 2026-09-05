from __future__ import annotations

import base64
import builtins
import hashlib
import json
from pathlib import Path

import httpx
import pytest

from llmbench import runner
from llmbench.artifacts import RunArtifactWriter
from llmbench.client import OpenAICompatibleClient
from llmbench.datasets import load_dataset
from llmbench.images import prepare_image_messages
from llmbench.repro import build_run_manifest, canonical_hash
from llmbench.runspec import RunSpec
from llmbench.schemas import RequestResult
from llmbench.session import ResumeMismatch

APIS = ("chat", "responses", "messages", "generate-content")
ANSWER = '{"answer":"A"}'


@pytest.fixture
def visual_dataset(tmp_path):
    pillow = pytest.importorskip("PIL.Image")
    directory = tmp_path / "dataset"
    directory.mkdir()
    assets = directory / "assets"
    assets.mkdir()
    paths = [assets / "first.png", assets / "second.png"]
    for path, color in zip(paths, ["white", "black"], strict=True):
        pillow.new("RGB", (3, 2), color).save(path)
    record = {
        "id": "visual-1",
        "dataset": "custom-vision",
        "type": "multiple_choice",
        "question": "Which colors appear in the first and second images?",
        "choices": {"A": "White then black", "B": "Black then white"},
        "answer": "A",
        "metadata": {"assets": ["assets/first.png", "assets/second.png"]},
    }
    dataset = directory / "questions.jsonl"
    dataset.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return dataset, paths


def make_spec(dataset: Path, output: Path, *, api="chat", stream=False, n_samples=2):
    return RunSpec(
        base_url="http://image.test/v1",
        api_key="",
        api=api,
        model="fixture-vision",
        dataset=(str(dataset),),
        limit=None,
        n_samples=n_samples,
        concurrency=1,
        temperature=0,
        top_p=1,
        max_tokens=4096,
        stream=stream,
        retries=0,
        output_dir=output,
    )


def response_payload(api):
    if api == "chat":
        return {
            "choices": [{"message": {"content": ANSWER}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 24, "completion_tokens": 6},
        }
    if api == "responses":
        return {
            "status": "completed",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": ANSWER}]}],
            "usage": {"input_tokens": 24, "output_tokens": 6},
        }
    if api == "messages":
        return {
            "content": [{"type": "text", "text": ANSWER}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 24, "output_tokens": 6},
        }
    return {
        "candidates": [{"content": {"parts": [{"text": ANSWER}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 24, "candidatesTokenCount": 6},
    }


def response_stream(api):
    payload = response_payload(api)
    if api == "chat":
        frames = [
            {"choices": [{"delta": {"content": ANSWER}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            {"choices": [], "usage": payload["usage"]},
            "[DONE]",
        ]
    elif api == "responses":
        frames = [
            {"type": "response.output_text.delta", "delta": ANSWER},
            {"type": "response.completed", "response": payload},
        ]
    elif api == "messages":
        frames = [
            {"type": "message_start", "message": {"usage": {"input_tokens": 24}}},
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": ANSWER}},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 6},
            },
            {"type": "message_stop"},
        ]
    else:
        frames = [payload]
    return "".join(
        f"data: {frame if isinstance(frame, str) else json.dumps(frame)}\n\n" for frame in frames
    )


def mock_client_factory(monkeypatch, calls, *, after_post=None):
    def factory(spec):
        def handle(request):
            assert request.method == "POST", "evaluation must not discover models"
            calls.append(json.loads(request.content))
            if after_post:
                after_post(len(calls))
            if spec.stream:
                return httpx.Response(
                    200,
                    text=response_stream(spec.api),
                    headers={"content-type": "text/event-stream"},
                )
            return httpx.Response(200, json=response_payload(spec.api))

        return OpenAICompatibleClient(
            base_url=spec.base_url,
            api_key="",
            api=spec.api,
            retries=0,
            transport=httpx.MockTransport(handle),
        )

    monkeypatch.setattr(runner, "_client", factory)


def assert_native_images(body, api, expected_bytes):
    if api == "generate-content":
        content = body["contents"][0]["parts"]
    elif api == "responses":
        content = body["input"][0]["content"]
    else:
        content = body["messages"][0]["content"]
    assert len(content) == 3
    text = content[0]["text"]
    assert "Which colors appear" in text
    assert "A. White then black" in text
    assert "B. Black then white" in text
    assert "Return JSON with only the answer field" in text
    for block, raw in zip(content[1:], expected_bytes, strict=True):
        if api == "chat":
            assert block["type"] == "image_url"
            url = block["image_url"]["url"]
            assert url.startswith("data:image/png;base64,")
            encoded = url.partition(",")[2]
        elif api == "responses":
            assert block["type"] == "input_image"
            assert block["image_url"].startswith("data:image/png;base64,")
            encoded = block["image_url"].partition(",")[2]
        elif api == "messages":
            assert block["type"] == "image"
            assert block["source"]["type"] == "base64"
            assert block["source"]["media_type"] == "image/png"
            encoded = block["source"]["data"]
        else:
            assert block["inlineData"]["mimeType"] == "image/png"
            encoded = block["inlineData"]["data"]
        assert base64.b64decode(encoded, validate=True) == raw


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize("stream", [False, True])
def test_scored_image_run_all_protocols_persists_hashes_and_samples(
    tmp_path, visual_dataset, monkeypatch, api, stream
):
    dataset, image_paths = visual_dataset
    calls = []
    mock_client_factory(monkeypatch, calls)
    spec = make_spec(dataset, tmp_path / "run", api=api, stream=stream)
    items = load_dataset(str(dataset))
    assert items[0].is_image
    assert items[0].metadata["asset_base_dir"] == str(dataset.parent.resolve())
    canonical_messages, assets = prepare_image_messages(items[0])
    summary, paths = runner.run_evaluation(spec, mode="run")
    assert len(calls) == 2
    for body in calls:
        assert_native_images(body, api, [path.read_bytes() for path in image_paths])
    assert summary["evaluation"]["complete"] is True
    assert summary["evaluation"]["completed_requests"] == 2
    assert summary["quality"]["sample_mean_score"] == 1
    assert summary["quality"]["quality_valid"] is True
    rows = [json.loads(line) for line in paths["raw"].read_text().splitlines()]
    assert {row["sample_id"] for row in rows} == {1, 2}
    for row in rows:
        assert row["score"] == 1
        assert row["parsed_answer"] == "A"
        assert row["error"] is None
        assert row["images"] == assets
        assert RequestResult.from_dict(row).images == assets
    persisted = RunArtifactWriter(spec.output_dir).existing_results()
    assert [row.images for row in persisted] == [assets, assets]
    manifest = json.loads((spec.output_dir / "run_manifest.json").read_text())
    assert manifest["image_inputs"] == [
        {
            "dataset": "custom-vision",
            "question_id": "visual-1",
            "messages_sha256": canonical_hash(canonical_messages),
            "assets": assets,
        }
    ]
    for path in spec.output_dir.iterdir():
        if path.is_file():
            rendered = path.read_text()
            assert "data:image/" not in rendered
            for image_path in image_paths:
                assert base64.b64encode(image_path.read_bytes()).decode() not in rendered
    assert all(asset["sha256"] in paths["raw"].read_text() for asset in assets)


def test_image_byte_change_invalidates_resume_without_dataset_file_change(
    tmp_path, visual_dataset, monkeypatch
):
    dataset, image_paths = visual_dataset
    calls = []
    mock_client_factory(monkeypatch, calls)
    spec = make_spec(dataset, tmp_path / "resume")
    runner.run_evaluation(spec, mode="run")
    dataset_digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    initial_manifest = json.loads((spec.output_dir / "run_manifest.json").read_text())
    raw_results = (spec.output_dir / "raw_results.jsonl").read_bytes()
    # A completed unchanged resume makes no extra requests and preserves image metadata.
    resumed, _ = runner.run_evaluation(spec, mode="run", resume=True)
    assert resumed["evaluation"]["completed_requests"] == 2
    assert len(calls) == 2
    pillow = pytest.importorskip("PIL.Image")
    pillow.new("RGB", (3, 2), "red").save(image_paths[0])
    assert hashlib.sha256(dataset.read_bytes()).hexdigest() == dataset_digest
    updated_manifest = build_run_manifest(
        run_id=initial_manifest["run_id"],
        mode="run",
        model=spec.model,
        base_url=spec.base_url,
        config=initial_manifest["config"],
        items=load_dataset(str(dataset)),
        n_samples=spec.n_samples,
    )
    assert updated_manifest["fingerprint"] != initial_manifest["fingerprint"]
    assert updated_manifest["prompts_sha256"] != initial_manifest["prompts_sha256"]
    assert updated_manifest["question_keys_sha256"] == initial_manifest["question_keys_sha256"]
    with pytest.raises(ResumeMismatch, match="does not match"):
        runner.run_evaluation(spec, mode="run", resume=True)
    assert len(calls) == 2
    assert (spec.output_dir / "raw_results.jsonl").read_bytes() == raw_results


@pytest.mark.parametrize("failure", ["missing", "corrupt", "missing-extra"])
def test_image_preflight_failures_happen_before_any_post(
    tmp_path, visual_dataset, monkeypatch, failure
):
    dataset, image_paths = visual_dataset
    calls = []
    mock_client_factory(monkeypatch, calls)
    spec = make_spec(dataset, tmp_path / "invalid")
    if failure == "missing":
        image_paths[1].unlink()
        expected = "cannot read image asset"
    elif failure == "corrupt":
        image_paths[1].write_bytes(b"not a PNG")
        expected = "invalid image asset"
    else:
        original_import = builtins.__import__

        def no_pillow(name, *args, **kwargs):
            if name == "PIL" or name.startswith("PIL."):
                raise ImportError("image extra deliberately unavailable")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_pillow)
        expected = r"install 'llm-bench\[image\]'"
    with pytest.raises(ValueError, match=expected):
        runner.run_evaluation(spec, mode="run")
    assert calls == []
    assert not (spec.output_dir / "raw_results.jsonl").exists()


def test_image_change_between_samples_is_rejected_before_second_post(
    tmp_path, visual_dataset, monkeypatch
):
    dataset, image_paths = visual_dataset
    pillow = pytest.importorskip("PIL.Image")
    calls = []

    def replace_after_first_post(count):
        if count == 1:
            pillow.new("RGB", (3, 2), "red").save(image_paths[0])

    mock_client_factory(monkeypatch, calls, after_post=replace_after_first_post)
    spec = make_spec(dataset, tmp_path / "changed-during-run")
    with pytest.raises(ValueError, match=r"image.*changed|changed.*image"):
        runner.run_evaluation(spec, mode="run")
    assert len(calls) == 1
    rows = RunArtifactWriter(spec.output_dir).existing_results()
    assert len(rows) == 1
    assert rows[0].score == 1


@pytest.mark.parametrize("tampered", [False, True])
def test_declared_image_hash_rejects_replaced_valid_png_before_post(
    tmp_path, visual_dataset, monkeypatch, tampered
):
    dataset, image_paths = visual_dataset
    record = json.loads(dataset.read_text())
    record["metadata"]["asset_sha256"] = {
        asset: hashlib.sha256(path.read_bytes()).hexdigest()
        for asset, path in zip(record["metadata"]["assets"], image_paths, strict=True)
    }
    dataset.write_text(json.dumps(record) + "\n", encoding="utf-8")
    if tampered:
        pillow = pytest.importorskip("PIL.Image")
        pillow.new("RGB", (3, 2), "red").save(image_paths[1])
    calls = []
    mock_client_factory(monkeypatch, calls)
    spec = make_spec(dataset, tmp_path / "declared-hash", n_samples=1)
    if tampered:
        with pytest.raises(ValueError, match="image asset SHA-256 does not match"):
            runner.run_evaluation(spec, mode="run")
        assert calls == []
        assert not (spec.output_dir / "raw_results.jsonl").exists()
    else:
        summary, _ = runner.run_evaluation(spec, mode="run")
        assert len(calls) == 1
        assert summary["quality"]["sample_mean_score"] == 1
