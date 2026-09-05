"""Build the PinchBench pack: the official OpenClaw agent task descriptors."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import re

from packbuild import Pack, build, github_tree

REVISION = "97118e1f6bb1e8993458f84ed85927f5e328f174"
REPO = "Studio-Intrinsic/clawbench"
SOURCE = f"https://api.github.com/repos/{REPO}/git/trees/{REVISION}?recursive=1"

PACK = Pack(
    dataset="pinchbench",
    package="llmbench_data_pinchbench",
    revision=REVISION,
    source=SOURCE,
    type="agent",
    category="OpenClaw智能体能力综合测评",
    metric="success_rate",
    license="MIT",
    restriction="requires remote OpenClaw executor and task-specific accounts",
    recommended_max_tokens=8192,
    flags={"official_full_set": True},
    item_metadata={
        "capability": "agent",
        "adapter": "official_harness",
        "executor_image": f"quanttrio/llmbench-pinchbench:{REVISION[:12]}",
        "executor_command": ...,
        "network": True,
    },
    max_tokens_in_record=False,
)


def convert(pack: Pack) -> list[dict]:
    paths = sorted(
        path
        for item in github_tree(REPO, REVISION)
        if (path := item["path"]).startswith("tasks/task_")
        and path.endswith(".md")
        and "TEMPLATE" not in path
    )
    records = []
    for path in paths:
        task_id = re.sub(r"\.md$", "", path.split("/", 1)[1])
        records.append(
            pack.record(
                id=f"pinchbench-{task_id}",
                question=f"Run official PinchBench task: {task_id}",
                metadata={"task_id": task_id},
                late_metadata={"executor_command": ["run-task", task_id]},
            )
        )
    return records


if __name__ == "__main__":
    build(PACK, convert, script=__file__)
