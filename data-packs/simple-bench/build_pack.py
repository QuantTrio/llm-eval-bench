"""Build the SimpleBench pack from the public 10-question release."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import re

from packbuild import Pack, build, github_file

REVISION = "fbc2e429085bdedad7d1a236d2bc9bc18c95f16e"
REPO = "simple-bench/SimpleBench"
SOURCE = f"https://api.github.com/repos/{REPO}/contents/simple_bench_public.json?ref={REVISION}"

PACK = Pack(
    dataset="simple-bench",
    package="llmbench_data_simple_bench",
    revision=REVISION,
    source=SOURCE,
    type="multiple_choice",
    category="常识推理",
    metric="accuracy",
    license="MIT",
    flags={"official_public_full_set": True},
)


def convert(pack: Pack) -> list[dict]:
    payload = json.loads(github_file(REPO, "simple_bench_public.json", REVISION))
    records = []
    for row in payload["eval_data"]:
        parts = re.split(r"\n([A-F])\.\s+", row["prompt"])
        records.append(
            pack.record(
                id=f"simple-bench-{row['question_id']}",
                question=parts[0].strip(),
                choices={
                    parts[index]: parts[index + 1].strip() for index in range(1, len(parts), 2)
                },
                answer=row["answer"],
            )
        )
    return records


if __name__ == "__main__":
    build(PACK, convert, script=__file__)
