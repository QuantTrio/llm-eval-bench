from __future__ import annotations

import hashlib
import json
import tempfile
import time
import urllib.request
from collections import defaultdict, deque
from pathlib import Path

import pyarrow.parquet as pq

REVISION = "5234c61e804e85e5994b14b002fa9ec34487bcfb"
CONFIGS = ("boolq", "cb", "copa", "rte", "wic", "wsc.fixed")
ROOT = f"https://huggingface.co/datasets/aps/super_glue/resolve/{REVISION}"
PACKAGE = Path(__file__).parent / "llmbench_data_superglue"


def download(url: str) -> bytes:
    error = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                return response.read()
        except OSError as exc:
            error = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"failed to download after retries: {url}: {error}")


def convert(config: str, row: dict) -> tuple[str, dict[str, str], str]:
    label = int(row["label"])
    if config == "boolq":
        return (
            f"Passage: {row['passage']}\nQuestion: {row['question']}",
            {"A": "False", "B": "True"},
            "B" if label else "A",
        )
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
        question = (
            f"Word: {row['word']}\nSentence 1: {row['sentence1']}\n"
            f"Sentence 2: {row['sentence2']}\nDoes the word have the same meaning?"
        )
        return question, {"A": "False", "B": "True"}, "B" if label else "A"
    question = f"Text: {row['text']}\nDoes '{row['span2_text']}' refer to '{row['span1_text']}'?"
    return question, {"A": "False", "B": "True"}, "B" if label else "A"


def main() -> None:
    groups = defaultdict(deque)
    for config in CONFIGS:
        path = f"{config}/validation-00000-of-00001.parquet"
        payload = download(f"{ROOT}/{path}")
        with tempfile.NamedTemporaryFile(suffix=".parquet") as handle:
            handle.write(payload)
            handle.flush()
            groups[config].extend(pq.read_table(handle.name).to_pylist())
    selected = []
    while len(selected) < 500 and groups:
        for config in sorted(list(groups)):
            selected.append((config, groups[config].popleft()))
            if not groups[config]:
                del groups[config]
            if len(selected) == 500:
                break
    records = []
    for index, (config, row) in enumerate(selected):
        question, choices, answer = convert(config, row)
        records.append(
            {
                "id": f"superglue-{config}-{index:04d}",
                "dataset": "superglue",
                "subset": config,
                "type": "multiple_choice",
                "question": question,
                "choices": choices,
                "answer": answer,
                "metadata": {
                    "benchmark_category": "自然语言理解",
                    "benchmark_metric": "accuracy",
                    "recommended_max_tokens": 4096,
                    "regression_subset": True,
                },
            }
        )
    output = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in records
    ).encode()
    (PACKAGE / "superglue.jsonl").write_bytes(output)
    manifest = {
        "name": "quanttrio-llmbench-data-superglue",
        "version": "0.5.0",
        "package": "llmbench_data_superglue",
        "source_revision": REVISION,
        "datasets": {
            "superglue": {
                "file": "superglue.jsonl",
                "count": len(records),
                "type": "multiple_choice",
                "category": "自然语言理解",
                "metric": "accuracy",
                "license": "LicenseRef-SuperGLUE-Mixed",
                "restriction": "local build only",
                "source": ROOT,
                "sha256": hashlib.sha256(output).hexdigest(),
                "recommended_max_tokens": 4096,
                "regression_subset": True,
            }
        },
    }
    (PACKAGE / "pack.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(records)} local-only records")


if __name__ == "__main__":
    main()
