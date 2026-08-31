from __future__ import annotations

import base64
import hashlib
import json
import tempfile
import time
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

REVISION = "2f069730be515ea60778413777816b53e2d2a697"
TASK_SOURCE = f"https://huggingface.co/datasets/TIGER-Lab/MMEB-eval/resolve/{REVISION}/CIFAR-100/test-00000-of-00001.parquet"
CIFAR_REVISION = "aadb3af77e9048adbea6b47c21a81e47dd092ae5"
CIFAR_SOURCE = (
    "https://huggingface.co/datasets/uoft-cs/cifar100/resolve/"
    f"{CIFAR_REVISION}/cifar100/test-00000-of-00001.parquet"
)
PACKAGE = Path(__file__).parent / "llmbench_data_mmeb_v2_image"


def download(url: str, timeout: int) -> bytes:
    error = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return response.read()
        except OSError as exc:
            error = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"failed to download after retries: {url}: {error}")


def main() -> None:
    task_payload = download(TASK_SOURCE, 120)
    with tempfile.NamedTemporaryFile(suffix=".parquet") as handle:
        handle.write(task_payload)
        handle.flush()
        tasks = pq.read_table(handle.name).to_pylist()[:100]
    cifar_payload = download(CIFAR_SOURCE, 300)
    with tempfile.NamedTemporaryFile(suffix=".parquet") as handle:
        handle.write(cifar_payload)
        handle.flush()
        images = pq.read_table(handle.name).to_pylist()[:100]
    records = []
    for index, task in enumerate(tasks):
        image = "data:image/png;base64," + base64.b64encode(images[index]["img"]["bytes"]).decode()
        targets = [str(value) for value in task["tgt_text"]]
        records.append(
            {
                "id": f"mmeb-v2-image-{index:04d}",
                "dataset": "mmeb-v2-image",
                "type": "embedding",
                "question": image,
                "answer": targets[0],
                "metadata": {
                    "positive": targets[0],
                    "negatives": targets[1:10],
                    "benchmark_category": "图像向量嵌入",
                    "benchmark_metric": "recall_at_1",
                    "capability": "embedding",
                    "adapter": "embedding",
                    "recommended_max_tokens": 1,
                    "regression_subset": True,
                },
            }
        )
    output = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in records
    ).encode()
    (PACKAGE / "mmeb-v2-image.jsonl").write_bytes(output)
    manifest = {
        "name": "quanttrio-llmbench-data-mmeb-v2-image",
        "version": "0.5.0",
        "package": "llmbench_data_mmeb_v2_image",
        "source_revision": REVISION,
        "datasets": {
            "mmeb-v2-image": {
                "file": "mmeb-v2-image.jsonl",
                "count": len(records),
                "type": "embedding",
                "category": "图像向量嵌入",
                "metric": "recall_at_1",
                "license": "LicenseRef-MMEB-CIFAR-Upstream",
                "restriction": "local build only",
                "source": TASK_SOURCE,
                "sha256": hashlib.sha256(output).hexdigest(),
                "recommended_max_tokens": 1,
                "regression_subset": True,
            }
        },
    }
    (PACKAGE / "pack.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(records)} local-only image embedding records")


if __name__ == "__main__":
    main()
