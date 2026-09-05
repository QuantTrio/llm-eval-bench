from __future__ import annotations

import base64
import hashlib
import json
import runpy
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "bundle_image_data.py"
bundle_image_data = runpy.run_path(str(SCRIPT))["bundle_image_data"]
# A valid one-pixel PNG, kept as bytes so conversion requires no optional Pillow.
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a9xoAAAAASUVORK5CYII="
)


def _source(tmp_path: Path, *, asset: str | None = None) -> tuple[Path, list[dict]]:
    inline = asset or "data:image/png;base64," + base64.b64encode(PNG).decode("ascii")
    rows = [
        {
            "id": "mmmu-choice",
            "dataset": "mmmu",
            "type": "multiple_choice",
            "question": "Compare <image 1> and <image 2>.",
            "choices": {"A": "same", "B": "different"},
            "answer": "A",
            "metadata": {"assets": [inline, inline], "difficulty": "Easy"},
        },
        {
            "id": "mmmu-open",
            "dataset": "mmmu",
            "type": "multiple_choice",
            "question": "What number is shown in <image 1>?",
            "choices": {},
            "answer": "1",
            "metadata": {"assets": [inline]},
        },
    ]
    source = tmp_path / "input.jsonl"
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    source.write_bytes(payload)
    (tmp_path / "pack.json").write_text(
        json.dumps(
            {
                "name": "fixture-mmmu",
                "version": "0.5.0",
                "source_revision": "fixture-revision",
                "datasets": {
                    "mmmu": {
                        "file": "input.jsonl",
                        "count": 2,
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "license": "Apache-2.0",
                        "source": "fixture source",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return source, rows


def test_conversion_preserves_rows_deduplicates_exact_png_and_is_repeatable(tmp_path) -> None:
    source, originals = _source(tmp_path)
    output = tmp_path / "core" / "data"
    output.mkdir(parents=True)
    text = output / "text.jsonl"
    text.write_bytes(b'{"id":"existing"}\n')
    (output / "manifest.json").write_text('{"text":{"count":1}}\n', encoding="utf-8")
    metadata = bundle_image_data(source, output)
    rows = [json.loads(line) for line in (output / "mmmu.jsonl").read_bytes().splitlines()]
    assert [row["id"] for row in rows] == [row["id"] for row in originals]
    assert [row["answer"] for row in rows] == [row["answer"] for row in originals]
    assert [row["question"] for row in rows] == [row["question"] for row in originals]
    assert [row["choices"] for row in rows] == [row["choices"] for row in originals]
    assert [row["type"] for row in rows] == ["multiple_choice", "mmmu_open"]
    digest = hashlib.sha256(PNG).hexdigest()
    asset = f"data/images/mmmu/{digest}.png"
    assert rows[0]["metadata"]["assets"] == [asset, asset]
    assert rows[1]["metadata"]["assets"] == [asset]
    assert rows[0]["metadata"]["asset_sha256"] == {asset: digest}
    assert rows[0]["metadata"]["difficulty"] == "Easy"
    assert (output.parent / asset).read_bytes() == PNG
    assert metadata["image_count"] == 1
    assert metadata["image_reference_count"] == 3
    assert metadata["source_revision"] == "fixture-revision"
    assert metadata["source"] == "fixture source"
    assert metadata["license"] == "Apache-2.0"
    assert json.loads((output / "manifest.json").read_text())["text"] == {"count": 1}
    assert text.read_bytes() == b'{"id":"existing"}\n'
    before = {
        p.relative_to(output): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in output.rglob("*")
        if p.is_file()
    }
    assert bundle_image_data(source, output) == metadata
    after = {
        p.relative_to(output): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in output.rglob("*")
        if p.is_file()
    }
    assert before == after


@pytest.mark.parametrize("asset", ["https://example.com/image.png", "data:image/png;base64,?bad"])
def test_conversion_rejects_nonlocal_or_invalid_assets_without_output(tmp_path, asset) -> None:
    source, _ = _source(tmp_path, asset=asset)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="PNG"):
        bundle_image_data(source, output)
    assert not output.exists()


def test_conversion_checks_source_provenance_before_writing(tmp_path) -> None:
    source, _ = _source(tmp_path)
    source.write_bytes(source.read_bytes() + b"\n")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="SHA-256"):
        bundle_image_data(source, output)
    assert not output.exists()


def test_conversion_does_not_overwrite_conflicting_existing_assets(tmp_path) -> None:
    source, _ = _source(tmp_path)
    output = tmp_path / "output"
    bundle_image_data(source, output)
    image = next((output / "images" / "mmmu").iterdir())
    image.write_bytes(b"existing different content")
    before = (output / "manifest.json").read_bytes()
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        bundle_image_data(source, output)
    assert image.read_bytes() == b"existing different content"
    assert (output / "manifest.json").read_bytes() == before
