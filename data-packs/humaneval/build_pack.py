"""Build the HumanEval pack: the first 100 problems, scored in a remote sandbox."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from packbuild import Pack, build, read_jsonl_gz

REVISION = "6d43fb980f9fee3c892a914eda09951f772ad10d"
SOURCE = f"https://raw.githubusercontent.com/openai/human-eval/{REVISION}/data/HumanEval.jsonl.gz"

PACK = Pack(
    dataset="humaneval",
    package="llmbench_data_humaneval",
    revision=REVISION,
    source=SOURCE,
    type="code",
    category="代码能力",
    metric="pass_at_1",
    license="MIT",
    limit=100,
    flags={"regression_subset": True},
    item_metadata={"capability": "agent", "adapter": "remote_executor"},
)


def convert(pack: Pack) -> list[dict]:
    return [
        pack.record(
            id=row["task_id"],
            question=row["prompt"],
            answer=row["canonical_solution"],
            metadata={"entry_point": row["entry_point"], "test": row["test"]},
        )
        for row in read_jsonl_gz(SOURCE)[: PACK.limit]
    ]


if __name__ == "__main__":
    build(PACK, convert, script=__file__)
