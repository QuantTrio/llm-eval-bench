"""Build the Terminal-Bench 2.0 pack: the official 89 task descriptors."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from packbuild import Pack, build, github_tree

REVISION = "2fd12b88aafdd04a52c298e3940bcb189f9766d6"
REPO = "harbor-framework/terminal-bench-2"
SOURCE = f"https://api.github.com/repos/{REPO}/git/trees/{REVISION}?recursive=1"

PACK = Pack(
    dataset="terminal-bench-2",
    package="llmbench_data_terminal_bench_2",
    revision=REVISION,
    source=SOURCE,
    type="agent",
    category="AI Agent - 工具使用",
    metric="success_rate",
    license="Apache-2.0",
    restriction="requires version-matched remote executor image",
    recommended_max_tokens=8192,
    flags={"official_full_set": True},
    item_metadata={
        "capability": "agent",
        "adapter": "official_harness",
        "executor_image": f"quanttrio/llmbench-terminal-bench-2:{REVISION[:12]}",
        "executor_command": ...,
        "network": False,
    },
    max_tokens_in_record=False,
)


def convert(pack: Pack) -> list[dict]:
    task_ids = sorted(
        item["path"].split("/", 1)[0]
        for item in github_tree(REPO, REVISION)
        if item["path"].count("/") == 1 and item["path"].endswith("/task.toml")
    )
    return [
        pack.record(
            id=f"terminal-bench-2-{task_id}",
            question=f"Run official Terminal-Bench 2.0 task: {task_id}",
            metadata={"task_id": task_id},
            late_metadata={"executor_command": ["run-task", task_id]},
        )
        for task_id in task_ids
    ]


if __name__ == "__main__":
    build(PACK, convert, script=__file__)
