"""Build the GDPval gold pack: 200 task descriptors balanced across occupations."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from packbuild import Pack, balanced, build, group_by, read_parquet

REVISION = "11e7900cdcac61bc4daf59e65feb238acda98fbf"
SOURCE = (
    "https://huggingface.co/datasets/openai/gdpval/resolve/"
    f"{REVISION}/data/train-00000-of-00001.parquet"
)

PACK = Pack(
    dataset="gdpval-gold",
    package="llmbench_data_gdpval_gold",
    revision=REVISION,
    source=SOURCE,
    type="agent",
    category="生产力知识",
    metric="judge_score",
    license="LicenseRef-GDPval-Upstream",
    restriction="local build only",
    recommended_max_tokens=8192,
    limit=200,
    flags={"regression_subset": True},
    item_metadata={
        "capability": "agent",
        "adapter": "artifact_judge",
        "executor_image": f"quanttrio/llmbench-gdpval:{REVISION[:12]}",
        "executor_command": ...,
        "network": False,
    },
)


def convert(pack: Pack) -> list[dict]:
    groups = group_by(read_parquet(SOURCE, timeout=120), lambda row: str(row["occupation"]))
    return [
        pack.record(
            id=f"gdpval-{row['task_id']}",
            question=row["prompt"],
            metadata={
                "sector": row["sector"],
                "occupation": row["occupation"],
                "reference_files": row.get("reference_files") or [],
                "reference_file_urls": row.get("reference_file_urls") or [],
                "deliverable_files": row.get("deliverable_files") or [],
                "rubric": row.get("rubric_pretty"),
                "rubric_json": row.get("rubric_json"),
            },
            late_metadata={"executor_command": ["run-task", str(row["task_id"])]},
        )
        for _occupation, row in balanced(groups, PACK.limit)
    ]


if __name__ == "__main__":
    build(PACK, convert, script=__file__, noun="local-only descriptors")
