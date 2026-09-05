"""Build the IFBench pack: 200 prompts balanced across instruction types."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from packbuild import Pack, balanced, build, group_by, read_parquet

REVISION = "2e8a48de45ff3bf41242f927254ca81b59ca3ae2"
SOURCE = (
    "https://huggingface.co/datasets/allenai/IFBench_test/resolve/"
    f"{REVISION}/data/train-00000-of-00001.parquet"
)

PACK = Pack(
    dataset="ifbench",
    package="llmbench_data_ifbench",
    revision=REVISION,
    source=SOURCE,
    type="instruction",
    category="指令跟随",
    metric="prompt_loose_accuracy",
    license="ODC-By-1.0",
    restriction="Ai2 Responsible Use and third-party output terms",
    limit=200,
    flags={"regression_subset": True},
    item_metadata={"capability": "chat", "adapter": "official_verifier"},
)


def convert(pack: Pack) -> list[dict]:
    groups = group_by(
        read_parquet(SOURCE), lambda row: str((row.get("instruction_id_list") or ["unknown"])[0])
    )
    return [
        pack.record(
            id=f"ifbench-{row['key']}",
            question=row["prompt"],
            metadata={
                "instruction_id_list": row.get("instruction_id_list") or [],
                "kwargs": row.get("kwargs") or [],
            },
        )
        for _name, row in balanced(groups, PACK.limit)
    ]


if __name__ == "__main__":
    build(PACK, convert, script=__file__)
