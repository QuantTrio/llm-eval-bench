"""Build the MMMU pack: 500 items balanced across subjects, images inlined."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ast
import base64
import mimetypes
from collections import deque

from packbuild import Pack, balanced, build, read_json, read_parquet

REVISION = "98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68"
API = "https://huggingface.co/api/datasets/MMMU/MMMU"
ROOT = f"https://huggingface.co/datasets/MMMU/MMMU/resolve/{REVISION}"

PACK = Pack(
    dataset="mmmu",
    package="llmbench_data_mmmu",
    revision=REVISION,
    source=ROOT,
    type="multiple_choice",
    category="多模态理解",
    metric="accuracy",
    license="Apache-2.0",
    limit=500,
    flags={"regression_subset": True},
    item_metadata={"capability": "multimodal", "adapter": "multimodal_chat"},
)


def _assets(row: dict) -> list[str]:
    assets = []
    for index in range(1, 8):
        image = row.get(f"image_{index}")
        if not image or not image.get("bytes"):
            continue
        mime = mimetypes.guess_type(image.get("path") or "image.png")[0] or "image/png"
        assets.append(f"data:{mime};base64," + base64.b64encode(image["bytes"]).decode())
    return assets


def convert(pack: Pack) -> list[dict]:
    paths = sorted(
        item["rfilename"]
        for item in read_json(API)["siblings"]
        if item["rfilename"].endswith("/validation-00000-of-00001.parquet")
    )
    groups: dict[str, deque] = {}
    for path in paths:
        subject = path.split("/", 1)[0]
        groups.setdefault(subject, deque()).extend(read_parquet(f"{ROOT}/{path}", timeout=180))
    return [
        pack.record(
            id=f"mmmu-{row['id']}",
            subset=subject,
            question=row["question"],
            choices={
                chr(65 + index): str(value)
                for index, value in enumerate(ast.literal_eval(row["options"]))
            },
            answer=row["answer"],
            metadata={
                "assets": _assets(row),
                "difficulty": row.get("topic_difficulty"),
                "subfield": row.get("subfield"),
            },
        )
        for subject, row in balanced(groups, PACK.limit)
    ]


if __name__ == "__main__":
    build(PACK, convert, script=__file__)
