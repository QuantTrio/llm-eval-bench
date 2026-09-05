"""Build the SimpleQA pack: short factual questions graded by a judge model."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ast

from packbuild import Pack, build, read_csv

REVISION = "652c89d0ca9df547706735883097e9537d40dc47"
SOURCE = "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"

PACK = Pack(
    dataset="simpleqa",
    package="llmbench_data_simpleqa",
    revision=REVISION,
    source=SOURCE,
    type="judge",
    category="常识问答",
    metric="judge_accuracy",
    license="MIT",
    limit=500,
    flags={"regression_subset": True},
    item_metadata={"capability": "chat", "adapter": "judge"},
)


def convert(pack: Pack) -> list[dict]:
    records = []
    for index, row in enumerate(read_csv(SOURCE)[: PACK.limit]):
        upstream = ast.literal_eval(row["metadata"])
        records.append(
            pack.record(
                id=f"simpleqa-{index:04d}",
                question=row["problem"],
                answer=row["answer"],
                metadata={
                    "topic": upstream.get("topic"),
                    "answer_type": upstream.get("answer_type"),
                    "urls": upstream.get("urls") or [],
                },
                late_metadata={
                    "rubric": (
                        "Score 1 only when the response is factually equivalent to the "
                        f"reference answer {row['answer']!r}; otherwise score 0."
                    )
                },
            )
        )
    return records


if __name__ == "__main__":
    build(PACK, convert, script=__file__)
