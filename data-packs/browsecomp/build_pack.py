"""Build the BrowseComp pack. Questions and answers stay encrypted upstream."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from packbuild import Pack, build, read_csv

REVISION = "652c89d0ca9df547706735883097e9537d40dc47"
SOURCE = "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"

PACK = Pack(
    dataset="browsecomp",
    package="llmbench_data_browsecomp",
    revision=REVISION,
    source=SOURCE,
    type="agent",
    category="AI Agent - 信息收集",
    metric="judge_accuracy",
    license="MIT",
    restriction="encrypted; do not reveal questions or answers",
    recommended_max_tokens=8192,
    limit=100,
    flags={"regression_subset": True, "sensitive": True},
    item_metadata={"capability": "agent", "adapter": "remote_browser"},
)


def convert(pack: Pack) -> list[dict]:
    return [
        pack.record(
            id=f"browsecomp-{index:04d}",
            question="[encrypted BrowseComp task]",
            metadata={
                "encrypted_problem": row["problem"],
                "encrypted_answer": row["answer"],
                "canary": row["canary"],
            },
        )
        for index, row in enumerate(read_csv(SOURCE)[: PACK.limit])
    ]


if __name__ == "__main__":
    build(PACK, convert, script=__file__, noun="encrypted records")
