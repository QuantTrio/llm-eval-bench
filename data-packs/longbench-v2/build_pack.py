"""Build the LongBench v2 pack: 200 items balanced across domain and difficulty."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import os

from packbuild import Pack, balanced, build, group_by, read_json

REVISION = "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9"
SOURCE = f"https://huggingface.co/datasets/zai-org/LongBench-v2/resolve/{REVISION}/data.json"

PACK = Pack(
    dataset="longbench-v2",
    package="llmbench_data_longbench_v2",
    revision=REVISION,
    source=SOURCE,
    type="multiple_choice",
    category="长上下文能力",
    metric="accuracy",
    license="Apache-2.0",
    recommended_max_tokens=8192,
    limit=200,
    flags={"regression_subset": True},
)


def convert(pack: Pack) -> list[dict]:
    # The upstream file is multi-gigabyte; allow a pre-downloaded copy.
    local = os.environ.get("LONGBENCH_SOURCE_FILE")
    rows = (
        json.loads(Path(local).read_text(encoding="utf-8"))
        if local
        else read_json(SOURCE, timeout=600)
    )
    groups = group_by(rows, lambda row: (row.get("domain"), row.get("difficulty")))
    return [
        pack.record(
            id=f"longbench-v2-{row['_id']}",
            subset=row.get("domain"),
            question=f"Context:\n{row['context']}\n\nQuestion:\n{row['question']}",
            choices={key: row[f"choice_{key}"] for key in ("A", "B", "C", "D")},
            answer=row["answer"],
            metadata={
                "sub_domain": row.get("sub_domain"),
                "difficulty": row.get("difficulty"),
                "length": row.get("length"),
            },
        )
        for _group, row in balanced(groups, PACK.limit)
    ]


if __name__ == "__main__":
    build(PACK, convert, script=__file__)
