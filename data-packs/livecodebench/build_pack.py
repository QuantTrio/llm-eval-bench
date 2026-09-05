"""Build the LiveCodeBench pack: the 100 most recent contest problems."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from packbuild import Pack, build, read_jsonl

REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
SOURCE = (
    "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/"
    f"{REVISION}/test6.jsonl"
)

PACK = Pack(
    dataset="livecodebench",
    package="llmbench_data_livecodebench",
    revision=REVISION,
    source=SOURCE,
    type="agent",
    category="编程与软件工程",
    metric="pass_at_1",
    license="LicenseRef-LiveCodeBench-CC",
    restriction="local build only; review upstream terms",
    recommended_max_tokens=8192,
    limit=100,
    flags={"regression_subset": True},
    item_metadata={
        "capability": "agent",
        "adapter": "official_harness",
        "executor_image": f"quanttrio/llmbench-livecodebench:{REVISION[:12]}",
        "executor_command": ...,
        "network": False,
    },
)


def convert(pack: Pack) -> list[dict]:
    rows = sorted(
        read_jsonl(SOURCE, timeout=120),
        key=lambda row: (row.get("contest_date", ""), row["question_id"]),
        reverse=True,
    )[: PACK.limit]
    return [
        pack.record(
            id=f"livecodebench-{row['question_id']}",
            question=row["question_content"],
            metadata={
                "platform": row.get("platform"),
                "difficulty": row.get("difficulty"),
                "starter_code": row.get("starter_code"),
                "public_test_cases": row.get("public_test_cases"),
                "private_test_cases": row.get("private_test_cases"),
            },
            late_metadata={"executor_command": ["run-task", str(row["question_id"])]},
        )
        for row in rows
    ]


if __name__ == "__main__":
    build(PACK, convert, script=__file__)
