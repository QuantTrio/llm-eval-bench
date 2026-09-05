"""Build the SuperGLUE pack: 500 items balanced across its six tasks."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from collections import deque

from packbuild import Pack, balanced, build, read_parquet

REVISION = "5234c61e804e85e5994b14b002fa9ec34487bcfb"
ROOT = f"https://huggingface.co/datasets/aps/super_glue/resolve/{REVISION}"
CONFIGS = ("boolq", "cb", "copa", "rte", "wic", "wsc.fixed")

PACK = Pack(
    dataset="superglue",
    package="llmbench_data_superglue",
    revision=REVISION,
    source=ROOT,
    type="multiple_choice",
    category="自然语言理解",
    metric="accuracy",
    license="LicenseRef-SuperGLUE-Mixed",
    restriction="local build only",
    limit=500,
    flags={"regression_subset": True},
)


def _question(config: str, row: dict) -> tuple[str, dict[str, str], str]:
    """Each SuperGLUE task phrases its own prompt; the label is always an index."""
    label = int(row["label"])
    boolean = {"A": "False", "B": "True"}
    if config == "boolq":
        return f"Passage: {row['passage']}\nQuestion: {row['question']}", boolean, "BA"[not label]
    if config == "cb":
        return (
            f"Premise: {row['premise']}\nHypothesis: {row['hypothesis']}",
            {"A": "entailment", "B": "contradiction", "C": "neutral"},
            chr(65 + label),
        )
    if config == "copa":
        return (
            f"Premise: {row['premise']}\nWhat was the {row['question']}?",
            {"A": row["choice1"], "B": row["choice2"]},
            chr(65 + label),
        )
    if config == "rte":
        return (
            f"Premise: {row['premise']}\nHypothesis: {row['hypothesis']}",
            {"A": "entailment", "B": "not_entailment"},
            chr(65 + label),
        )
    if config == "wic":
        return (
            f"Word: {row['word']}\nSentence 1: {row['sentence1']}\n"
            f"Sentence 2: {row['sentence2']}\nDoes the word have the same meaning?",
            boolean,
            "BA"[not label],
        )
    return (
        f"Text: {row['text']}\nDoes '{row['span2_text']}' refer to '{row['span1_text']}'?",
        boolean,
        "BA"[not label],
    )


def convert(pack: Pack) -> list[dict]:
    groups = {
        config: deque(read_parquet(f"{ROOT}/{config}/validation-00000-of-00001.parquet"))
        for config in CONFIGS
    }
    records = []
    for index, (config, row) in enumerate(balanced(groups, PACK.limit)):
        question, choices, answer = _question(config, row)
        records.append(
            pack.record(
                id=f"superglue-{config}-{index:04d}",
                subset=config,
                question=question,
                choices=choices,
                answer=answer,
            )
        )
    return records


if __name__ == "__main__":
    build(PACK, convert, script=__file__, noun="local-only records")
