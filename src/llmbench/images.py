"""Bounded, offline image inputs. Pillow is loaded only for actual image requests."""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import re
import warnings
from importlib import resources
from pathlib import Path, PurePosixPath
from typing import Any

from .schemas import DatasetItem
from .scoring import build_prompt

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_TOTAL_IMAGE_BYTES = 64 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
MAX_IMAGES = 16
_IMAGE_MIMES = {"image/png", "image/jpeg", "image/webp", "image/gif"}
_DATA_URL_HEADER = re.compile(r"data:(image/[a-z0-9.+-]+);base64")


def parse_image_data_url(value: str) -> tuple[str, str]:
    """Validate a bounded canonical data URL without importing image dependencies."""
    if not isinstance(value, str):
        raise ValueError("image_url.url must be an image data URL string")
    header, separator, encoded = value.partition(",")
    match = _DATA_URL_HEADER.fullmatch(header)
    if not separator or not match or match[1] not in _IMAGE_MIMES:
        raise ValueError("image inputs require a PNG, JPEG, WebP or GIF base64 data URL")
    if not encoded or len(encoded) > 4 * ((MAX_IMAGE_BYTES + 2) // 3):
        raise ValueError("image is empty or exceeds the image byte limit")
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image data URL contains invalid base64") from exc
    if not decoded or len(decoded) > MAX_IMAGE_BYTES:
        raise ValueError("image is empty or exceeds the image byte limit")
    if base64.b64encode(decoded).decode("ascii") != encoded:
        raise ValueError("image data URL must use canonical base64 encoding")
    return match[1], encoded


def image_url_block(block: dict[str, Any]) -> tuple[str, str | None]:
    """Check the shared Chat-style image block; never coerce bytes into text."""
    if set(block) != {"type", "image_url"} or block.get("type") != "image_url":
        raise ValueError("malformed image_url content block")
    value = block["image_url"]
    if not isinstance(value, dict) or set(value) - {"url", "detail"} or "url" not in value:
        raise ValueError("image_url must contain url and optional detail")
    detail = value.get("detail")
    if "detail" in value and (not isinstance(detail, str) or detail not in {"auto", "low", "high"}):
        raise ValueError("image detail must be auto, low or high")
    parse_image_data_url(value["url"])
    return value["url"], detail


def _pillow_image() -> Any:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ValueError("install 'llm-bench[image]' to process image inputs") from exc
    return Image


def _read_asset(item: DatasetItem, asset: str) -> tuple[bytes, str, str | None]:
    if asset.startswith("data:"):
        mime, encoded = parse_image_data_url(asset)
        return base64.b64decode(encoded, validate=True), "data", mime
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", asset) or asset.startswith("//"):
        raise ValueError("offline image assets must be local files or package resources")
    package = item.metadata.get("resource_package")
    if package:
        relative = PurePosixPath(asset)
        if (
            not isinstance(package, str)
            or not asset
            or relative.is_absolute()
            or ".." in relative.parts
            or "\\" in asset
        ):
            raise ValueError("image package asset must stay inside its resource package")
        try:
            root = resources.files(package)
            target = root.joinpath(*relative.parts)
            if isinstance(root, Path) and isinstance(target, Path):
                target.resolve().relative_to(root.resolve())
        except (ImportError, TypeError, ValueError) as exc:
            raise ValueError("image package asset must stay inside its resource package") from exc
        source = f"{package}:{relative.as_posix()}"
    else:
        path = Path(asset).expanduser()
        if path.is_absolute():
            target = path.resolve()
        else:
            root = Path(item.metadata.get("asset_base_dir") or Path.cwd()).expanduser().resolve()
            target = (root / path).resolve()
            if not target.is_relative_to(root):
                raise ValueError("relative image asset must stay inside asset_base_dir")
        source = str(target)
    try:
        if not target.is_file():
            raise ValueError("image asset must be a regular file")
        with target.open("rb") as handle:
            raw = handle.read(MAX_IMAGE_BYTES + 1)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read image asset: {source}") from exc
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("image is empty or exceeds the image byte limit")
    return raw, source, None


def _prepare_asset(item: DatasetItem, asset: str, image_module: Any) -> tuple[str, dict, int]:
    raw, source, declared_mime = _read_asset(item, asset)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", image_module.DecompressionBombWarning)
            with image_module.open(io.BytesIO(raw)) as decoded:
                width, height = decoded.size
                mime = image_module.MIME.get(decoded.format)
                if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                    raise ValueError("image dimensions exceed the pixel limit")
                if mime not in _IMAGE_MIMES:
                    raise ValueError("unsupported multimodal image format")
                if getattr(decoded, "n_frames", 1) != 1:
                    raise ValueError("animated images are not supported")
                if declared_mime and mime != declared_mime:
                    raise ValueError("image data URL MIME type does not match decoded image")
                decoded.verify()
            # verify() checks container integrity; load() also detects truncated pixel data.
            with image_module.open(io.BytesIO(raw)) as decoded:
                decoded.load()
    except (
        OSError,
        SyntaxError,
        ValueError,
        image_module.DecompressionBombError,
        image_module.DecompressionBombWarning,
    ) as exc:
        raise ValueError(f"invalid image asset: {exc}") from exc
    digest = hashlib.sha256(raw).hexdigest()
    expected_hashes = item.metadata.get("asset_sha256") or {}
    if not isinstance(expected_hashes, dict):
        raise ValueError("asset_sha256 must map asset paths to hashes")
    expected = expected_hashes.get(asset)
    if expected is not None and digest != expected:
        raise ValueError("image asset SHA-256 does not match dataset metadata")
    metadata = {
        "sha256": digest,
        "mime_type": mime,
        "width": width,
        "height": height,
        "source": f"sha256:{digest}" if source == "data" else source,
    }
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{encoded}", metadata, len(raw)


def asset_data_url(item: DatasetItem, asset: str) -> str:
    """Read and verify one offline image, retaining the legacy adapter interface."""
    if not isinstance(asset, str) or not asset:
        raise ValueError("image asset must be a nonempty file path or data URL string")
    return _prepare_asset(item, asset, _pillow_image())[0]


def prepare_image_messages(item: DatasetItem) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build canonical messages and content hashes without changing image pixels.

    Dataset assets follow the complete scoring prompt. Custom canonical messages
    retain their block order. These two prompt sources cannot be combined.
    """
    assets = item.metadata.get("assets")
    if assets is None:
        assets = []
    if not isinstance(assets, list):
        raise ValueError("image assets must be a list of file paths or data URLs")
    if assets and item.messages:
        raise ValueError("image assets cannot be combined with prebuilt messages")
    if len(assets) > MAX_IMAGES:
        raise ValueError("image count exceeds the per-item limit")
    metadata: list[dict[str, Any]] = []
    total_bytes = 0
    image_module = None

    def prepare(asset: str) -> str:
        nonlocal total_bytes, image_module
        if not isinstance(asset, str) or not asset:
            raise ValueError("image asset must be a nonempty file path or data URL string")
        if len(metadata) >= MAX_IMAGES:
            raise ValueError("image count exceeds the per-item limit")
        if image_module is None:
            image_module = _pillow_image()
        url, details, byte_count = _prepare_asset(item, asset, image_module)
        total_bytes += byte_count
        if total_bytes > MAX_TOTAL_IMAGE_BYTES:
            raise ValueError("images exceed the per-item total byte limit")
        metadata.append(details)
        return url

    if assets:
        content = [{"type": "text", "text": build_prompt(item)}]
        for asset in assets:
            content.append({"type": "image_url", "image_url": {"url": prepare(asset)}})
        return [{"role": "user", "content": content}], metadata
    if not item.messages:
        return [{"role": "user", "content": build_prompt(item)}], metadata
    messages = []
    for message in item.messages:
        if not isinstance(message, dict):
            raise ValueError("each message must be an object")
        content = message.get("content")
        if isinstance(content, str):
            messages.append(dict(message))
            continue
        if not isinstance(content, list):
            raise ValueError("message content must be text or canonical content blocks")
        blocks = []
        for block in content:
            if not isinstance(block, dict):
                raise ValueError("content block must be an object")
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                blocks.append(dict(block))
            elif block.get("type") == "image_url":
                url, detail = image_url_block(block)
                image_value = {"url": prepare(url)}
                if detail is not None:
                    image_value["detail"] = detail
                blocks.append({"type": "image_url", "image_url": image_value})
            else:
                raise ValueError("unsupported or malformed canonical content block")
        messages.append({**message, "content": blocks})
    return messages, metadata
