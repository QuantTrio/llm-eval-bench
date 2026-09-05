"""Build the AIME 2025 pack: the official 30-problem competition set."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from packbuild import Pack, build, read_parquet

REVISION = "a7b26602f8321cf71132eeb57c7f4d163e9b5a50"
SOURCE = (
    "https://huggingface.co/datasets/test-time-compute/aime_2025/resolve/"
    f"{REVISION}/data/test-00000-of-00001.parquet"
)

PACK = Pack(
    dataset="aime-2025",
    package="llmbench_data_aime_2025",
    revision=REVISION,
    source=SOURCE,
    type="math",
    category="数学推理",
    metric="exact_match",
    license="MIT",
    recommended_max_tokens=8192,
    flags={"official_full_set": True},
)


def convert(pack: Pack) -> list[dict]:
    records = []
    for index, row in enumerate(read_parquet(SOURCE)):
        upstream = row.get("metadata") or {}
        records.append(
            pack.record(
                id=f"aime-2025-{index + 1:02d}",
                question=row["question"],
                answer=str(row["answer"]),
                metadata={
                    "problem_type": upstream.get("problem_type"),
                    "difficulty": upstream.get("difficulty", "competition"),
                },
            )
        )
    return records


if __name__ == "__main__":
    build(PACK, convert, script=__file__)
