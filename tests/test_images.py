from __future__ import annotations

import base64
import builtins
import hashlib
import io
import json
import subprocess
import sys
from copy import deepcopy

import pytest

from llmbench import images
from llmbench.schemas import DatasetItem
from llmbench.scoring import build_prompt


@pytest.fixture
def image_bytes() -> bytes:
    pillow = pytest.importorskip("PIL.Image")
    output = io.BytesIO()
    pillow.new("RGB", (2, 3), "white").save(output, format="PNG")
    return output.getvalue()


def image_item(assets=None, **kwargs) -> DatasetItem:
    metadata = kwargs.pop("metadata", {})
    if assets is not None:
        metadata["assets"] = assets
    return DatasetItem(
        id="visual-1",
        dataset="mmmu",
        type="multiple_choice",
        question="What is shown?",
        answer="A",
        choices={"A": "White", "B": "Black"},
        metadata=metadata,
        **kwargs,
    )


def data_url(raw: bytes, mime="image/png") -> str:
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def test_image_messages_include_complete_prompt_and_safe_metadata(image_bytes) -> None:
    url = data_url(image_bytes)
    item = image_item([url, url])
    messages, metadata = images.prepare_image_messages(item)
    assert messages == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": build_prompt(item)},
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "image_url", "image_url": {"url": url}},
            ],
        }
    ]
    digest = hashlib.sha256(image_bytes).hexdigest()
    assert (
        metadata
        == [
            {
                "sha256": digest,
                "mime_type": "image/png",
                "width": 2,
                "height": 3,
                "source": f"sha256:{digest}",
            }
        ]
        * 2
    )
    assert "base64" not in json.dumps(metadata)
    assert "A. White" in messages[0]["content"][0]["text"]
    assert "Return JSON" in messages[0]["content"][0]["text"]


def test_prebuilt_images_keep_text_order_and_detail_without_mutation(image_bytes) -> None:
    url = data_url(image_bytes)
    original = [
        {"role": "system", "content": "You compare pictures."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "First:"},
                {"type": "image_url", "image_url": {"url": url, "detail": "high"}},
                {"type": "text", "text": "Compare with:"},
                {"type": "image_url", "image_url": {"url": url}},
            ],
        },
    ]
    item = image_item(messages=deepcopy(original))
    messages, metadata = images.prepare_image_messages(item)
    assert messages == original == item.messages
    assert len(metadata) == 2
    messages[1]["content"][1]["image_url"]["detail"] = "low"
    assert item.messages == original


def test_local_relative_and_absolute_images_use_actual_mime(tmp_path, image_bytes) -> None:
    path = tmp_path / "mislabelled.jpg"
    path.write_bytes(image_bytes)
    item = image_item([path.name], metadata={"asset_base_dir": str(tmp_path)})
    messages, metadata = images.prepare_image_messages(item)
    assert messages[0]["content"][1]["image_url"]["url"] == data_url(image_bytes)
    assert metadata[0]["mime_type"] == "image/png"
    assert metadata[0]["source"] == str(path.resolve())
    item.metadata["assets"] = [str(path)]
    assert images.prepare_image_messages(item)[1] == metadata


def test_hash_changes_when_same_file_contents_change(tmp_path, image_bytes) -> None:
    pillow = pytest.importorskip("PIL.Image")
    path = tmp_path / "image.png"
    path.write_bytes(image_bytes)
    item = image_item([str(path)])
    before = images.prepare_image_messages(item)[1]
    pillow.new("RGB", (2, 3), "black").save(path)
    after = images.prepare_image_messages(item)[1]
    assert before[0]["sha256"] != after[0]["sha256"]
    assert before[0]["source"] == after[0]["source"]


def test_package_resource_resolves_inside_package(tmp_path, image_bytes, monkeypatch) -> None:
    root = tmp_path / "package"
    path = root / "data" / "images" / "mmmu" / "image.png"
    path.parent.mkdir(parents=True)
    path.write_bytes(image_bytes)
    monkeypatch.setattr(images.resources, "files", lambda package: root)
    item = image_item(["data/images/mmmu/image.png"], metadata={"resource_package": "llmbench"})
    messages, metadata = images.prepare_image_messages(item)
    assert messages[0]["content"][1]["image_url"]["url"] == data_url(image_bytes)
    assert metadata[0]["source"] == "llmbench:data/images/mmmu/image.png"


@pytest.mark.parametrize(
    "asset", ["../outside.png", "/tmp/outside.png", "data/../../x.png", "a\\x.png"]
)
def test_package_traversal_rejected(asset, image_bytes) -> None:
    with pytest.raises(ValueError, match="inside its resource package"):
        images.prepare_image_messages(
            image_item([asset], metadata={"resource_package": "llmbench"})
        )


def test_local_and_package_symlink_escape_rejected(tmp_path, image_bytes, monkeypatch) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(image_bytes)
    (root / "link.png").symlink_to(outside)
    for asset in ["../outside.png", "link.png"]:
        with pytest.raises(ValueError, match="inside asset_base_dir"):
            images.prepare_image_messages(
                image_item([asset], metadata={"asset_base_dir": str(root)})
            )
    monkeypatch.setattr(images.resources, "files", lambda package: root)
    with pytest.raises(ValueError, match="inside its resource package"):
        images.prepare_image_messages(
            image_item(["link.png"], metadata={"resource_package": "llmbench"})
        )


@pytest.mark.parametrize(
    "asset", ["https://example.com/image.png", "http://x/a.png", "file:///x.png"]
)
def test_network_assets_are_rejected(asset, image_bytes) -> None:
    with pytest.raises(ValueError, match="offline image assets"):
        images.prepare_image_messages(image_item([asset]))


def test_missing_or_invalid_images_fail_before_request(tmp_path, image_bytes) -> None:
    with pytest.raises(ValueError, match="cannot read image asset"):
        images.prepare_image_messages(image_item([str(tmp_path / "missing.png")]))
    with pytest.raises(ValueError, match="invalid image asset"):
        images.prepare_image_messages(image_item([data_url(b"not really a PNG")]))
    with pytest.raises(ValueError, match="MIME type does not match"):
        images.prepare_image_messages(image_item([data_url(image_bytes, "image/jpeg")]))
    with pytest.raises(ValueError, match="invalid image asset"):
        images.prepare_image_messages(image_item([data_url(image_bytes[:-12])]))


@pytest.mark.parametrize(
    "format_name,mime",
    [("PNG", "image/png"), ("JPEG", "image/jpeg"), ("WEBP", "image/webp"), ("GIF", "image/gif")],
)
def test_supported_image_formats_keep_original_bytes(format_name, mime) -> None:
    pillow = pytest.importorskip("PIL.Image")
    buffer = io.BytesIO()
    pillow.new("RGB", (3, 2), "red").save(buffer, format=format_name)
    raw = buffer.getvalue()
    messages, metadata = images.prepare_image_messages(image_item([data_url(raw, mime)]))
    assert messages[0]["content"][1]["image_url"]["url"] == data_url(raw, mime)
    assert metadata[0]["mime_type"] == mime
    assert (metadata[0]["width"], metadata[0]["height"]) == (3, 2)


def test_animated_images_are_rejected() -> None:
    pillow = pytest.importorskip("PIL.Image")
    buffer = io.BytesIO()
    pillow.new("RGB", (2, 2), "red").save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=[pillow.new("RGB", (2, 2), "blue")],
    )
    with pytest.raises(ValueError, match="animated images are not supported"):
        images.prepare_image_messages(image_item([data_url(buffer.getvalue(), "image/gif")]))


@pytest.mark.parametrize(
    "value",
    [
        None,
        1,
        {},
        "https://example.com/image.png",
        "data:text/plain;base64,cG5n",
        "data:image/svg+xml;base64,cG5n",
        "data:image/png,cG5n",
        "data:image/png;base64,",
        "data:image/png;base64,not base64!!!",
        "data:image/png;base64,cG5n====",
    ],
)
def test_invalid_image_urls_fail_without_image_import(value) -> None:
    with pytest.raises(ValueError):
        images.parse_image_data_url(value)


def test_image_size_count_and_pixel_limits(image_bytes, monkeypatch) -> None:
    url = data_url(image_bytes)
    with pytest.raises(ValueError, match="image count"):
        images.prepare_image_messages(image_item([url] * (images.MAX_IMAGES + 1)))
    monkeypatch.setattr(images, "MAX_IMAGE_PIXELS", 5)
    with pytest.raises(ValueError, match="pixel limit"):
        images.prepare_image_messages(image_item([url]))
    monkeypatch.setattr(images, "MAX_IMAGE_PIXELS", 25_000_000)
    monkeypatch.setattr(images, "MAX_TOTAL_IMAGE_BYTES", len(image_bytes))
    with pytest.raises(ValueError, match="total byte limit"):
        images.prepare_image_messages(image_item([url, url]))
    monkeypatch.setattr(images, "MAX_IMAGE_BYTES", len(image_bytes) - 1)
    with pytest.raises(ValueError, match="byte limit"):
        images.prepare_image_messages(image_item([url]))


def test_missing_image_extra_is_clean_but_text_does_not_import_it(monkeypatch) -> None:
    original_import = builtins.__import__

    def no_pillow(name, *args, **kwargs):
        if name == "PIL" or name.startswith("PIL."):
            raise ImportError("Pillow deliberately unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_pillow)
    item = image_item()
    assert images.prepare_image_messages(item) == (
        [{"role": "user", "content": build_prompt(item)}],
        [],
    )
    item.messages = [{"role": "user", "content": "Text only."}]
    assert images.prepare_image_messages(item) == (item.messages, [])
    with pytest.raises(ValueError, match=r"install 'llm-bench\[image\]'"):
        images.prepare_image_messages(image_item(["missing.png"]))


def test_importing_text_request_modules_does_not_load_pillow() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from llmbench import adapters, protocols; "
            "assert not any(k == 'PIL' or k.startswith('PIL.') for k in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_conflicting_or_malformed_inputs_fail() -> None:
    with pytest.raises(ValueError, match="combined with prebuilt messages"):
        images.prepare_image_messages(
            image_item(["image.png"], messages=[{"role": "user", "content": "Different prompt"}])
        )
    for assets in ["image.png", {}, 1]:
        with pytest.raises(ValueError, match="assets must be a list"):
            images.prepare_image_messages(image_item(assets))
    for asset in [None, {}, 1, ""]:
        with pytest.raises(ValueError, match="image asset must be"):
            images.prepare_image_messages(image_item([asset]))
    for block in [None, {"type": "input_image"}, {"type": "text", "text": 1}]:
        with pytest.raises(ValueError):
            images.prepare_image_messages(
                image_item(messages=[{"role": "user", "content": [block]}])
            )
