"""Build the Aider Polyglot pack: 200 exercises, balanced across languages."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import re
from collections import deque

from packbuild import Pack, balanced, build, github_tree

REVISION = "7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f"
REPO = "Aider-AI/polyglot-benchmark"
SOURCE = f"https://api.github.com/repos/{REPO}/git/trees/{REVISION}?recursive=1"
PATTERN = re.compile(r"^([^/]+)/exercises/practice/([^/]+)/\.docs/instructions\.md$")

PACK = Pack(
    dataset="aider-polyglot",
    package="llmbench_data_aider_polyglot",
    revision=REVISION,
    source=SOURCE,
    type="agent",
    category="Agent能力评测",
    metric="pass_rate_1",
    license="LicenseRef-Upstream-Mixed",
    restriction="requires version-matched remote executor image",
    recommended_max_tokens=8192,
    limit=200,
    flags={"regression_subset": True},
    item_metadata={
        "capability": "agent",
        "adapter": "official_harness",
        "executor_image": f"quanttrio/llmbench-aider-polyglot:{REVISION[:12]}",
        "executor_command": ...,
        "network": False,
    },
    max_tokens_in_record=False,
)


def convert(pack: Pack) -> list[dict]:
    groups = {}
    for item in github_tree(REPO, REVISION):
        match = PATTERN.match(item["path"])
        if match:
            groups.setdefault(match.group(1), deque()).append(match.group(2))
    return [
        pack.record(
            id=f"aider-polyglot-{language}-{exercise}",
            subset=language,
            question=f"Run Aider Polyglot exercise {language}/{exercise}",
            metadata={"language": language, "exercise": exercise},
            late_metadata={"executor_command": ["run-task", language, exercise]},
        )
        for language, exercise in balanced(groups, PACK.limit)
    ]


if __name__ == "__main__":
    build(PACK, convert, script=__file__, noun="task descriptors")
