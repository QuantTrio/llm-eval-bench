"""Build the AGIEval pack: 200 questions taken round-robin across all 18 subsets."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import re
from collections import deque

from packbuild import Pack, balanced, build, github_file

REVISION = "84ab72d94318290aad2e4ec820d535a95a1f7552"
REPO = "ruixiangcui/AGIEval"
SUBSETS = (
    "aqua-rat",
    "gaokao-biology",
    "gaokao-chemistry",
    "gaokao-english",
    "gaokao-geography",
    "gaokao-history",
    "gaokao-mathqa",
    "gaokao-physics",
    "jec-qa-ca",
    "jec-qa-kd",
    "logiqa-en",
    "logiqa-zh",
    "lsat-ar",
    "lsat-lr",
    "lsat-rc",
    "sat-en-without-passage",
    "sat-en",
    "sat-math",
)

PACK = Pack(
    dataset="agieval",
    package="llmbench_data_agieval",
    revision=REVISION,
    source=f"https://github.com/{REPO}/tree/{REVISION}/data/v1_1",
    type="multiple_choice",
    category="综合能力",
    metric="accuracy",
    license="MIT",
    limit=200,
    flags={"regression_subset": True},
)


def convert(pack: Pack) -> list[dict]:
    groups = {}
    for subset in SUBSETS:
        text = github_file(REPO, f"data/v1_1/{subset}.jsonl", REVISION)
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        groups[subset] = deque(row for row in rows if row.get("options") and row.get("label"))
    records = []
    for index, (subset, row) in enumerate(balanced(groups, PACK.limit)):
        passage = str(row.get("passage") or "").strip()
        records.append(
            pack.record(
                id=f"agieval-{subset}-{index:04d}",
                subset=subset,
                question=(passage + "\n\n" if passage else "") + row["question"],
                choices={
                    chr(65 + offset): re.sub(r"^\([A-Z]\)\s*", "", str(option))
                    for offset, option in enumerate(row["options"])
                },
                answer=row["label"],
            )
        )
    return records


if __name__ == "__main__":
    build(PACK, convert, script=__file__)
