"""Build the Creative Writing v3 pack: the official prompts plus their rubric."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

from packbuild import Pack, build, github_file

REVISION = "c7c3ceef54c40a8ae02dc1c2e1a5e40970fe5c0b"
REPO = "EQ-bench/creative-writing-bench"

PACK = Pack(
    dataset="creative-writing-v3",
    package="llmbench_data_creative_writing_v3",
    revision=REVISION,
    source=f"https://github.com/{REPO}/tree/{REVISION}/data",
    type="judge",
    category="写作和创作",
    metric="judge_score",
    license="LicenseRef-Upstream-Unspecified",
    restriction="local build only",
    flags={"official_full_set": True},
    item_metadata={"capability": "chat", "adapter": "judge"},
)


def convert(pack: Pack) -> list[dict]:
    prompts = json.loads(github_file(REPO, "data/creative_writing_prompts_v3.json", REVISION))
    rubric = github_file(REPO, "data/creative_writing_criteria.txt", REVISION).strip()
    return [
        pack.record(
            id=f"creative-writing-v3-{int(key):02d}",
            question=row["writing_prompt"].replace(
                "<SEED>", (row.get("seed_modifiers") or [""])[0]
            ),
            metadata={
                "category": row.get("category"),
                "title": row.get("title"),
                "rubric": rubric,
            },
        )
        for key, row in sorted(prompts.items(), key=lambda pair: int(pair[0]))
    ]


if __name__ == "__main__":
    build(PACK, convert, script=__file__, noun="local-only prompts")
