"""Build the MMEB-v2 image pack: 100 CIFAR-100 retrieval items with inlined images."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import base64

from packbuild import Pack, build, read_parquet

REVISION = "2f069730be515ea60778413777816b53e2d2a697"
TASK_SOURCE = (
    "https://huggingface.co/datasets/TIGER-Lab/MMEB-eval/resolve/"
    f"{REVISION}/CIFAR-100/test-00000-of-00001.parquet"
)
CIFAR_REVISION = "aadb3af77e9048adbea6b47c21a81e47dd092ae5"
CIFAR_SOURCE = (
    "https://huggingface.co/datasets/uoft-cs/cifar100/resolve/"
    f"{CIFAR_REVISION}/cifar100/test-00000-of-00001.parquet"
)

PACK = Pack(
    dataset="mmeb-v2-image",
    package="llmbench_data_mmeb_v2_image",
    revision=REVISION,
    source=TASK_SOURCE,
    type="embedding",
    category="图像向量嵌入",
    metric="recall_at_1",
    license="LicenseRef-MMEB-CIFAR-Upstream",
    restriction="local build only",
    recommended_max_tokens=1,
    limit=100,
    flags={"regression_subset": True},
    item_metadata={"capability": "embedding", "adapter": "embedding"},
)


def convert(pack: Pack) -> list[dict]:
    tasks = read_parquet(TASK_SOURCE, timeout=120)[: PACK.limit]
    images = read_parquet(CIFAR_SOURCE, timeout=300)[: PACK.limit]
    records = []
    for index, task in enumerate(tasks):
        image = base64.b64encode(images[index]["img"]["bytes"]).decode()
        targets = [str(value) for value in task["tgt_text"]]
        records.append(
            pack.record(
                id=f"mmeb-v2-image-{index:04d}",
                question="data:image/png;base64," + image,
                answer=targets[0],
                metadata={"positive": targets[0], "negatives": targets[1:10]},
            )
        )
    return records


if __name__ == "__main__":
    build(PACK, convert, script=__file__)
