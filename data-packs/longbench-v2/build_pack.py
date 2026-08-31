from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from collections import defaultdict, deque
from pathlib import Path

REVISION = "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9"
SOURCE = f"https://huggingface.co/datasets/zai-org/LongBench-v2/resolve/{REVISION}/data.json"
PACKAGE = Path(__file__).parent / "llmbench_data_longbench_v2"


def main() -> None:
    source_file = os.environ.get("LONGBENCH_SOURCE_FILE")
    if source_file:
        rows = json.loads(Path(source_file).read_text(encoding="utf-8"))
    else:
        with urllib.request.urlopen(SOURCE, timeout=600) as response:
            rows = json.loads(response.read())
    groups = defaultdict(deque)
    for row in rows:
        groups[(row.get("domain"), row.get("difficulty"))].append(row)
    selected = []
    while len(selected) < 200 and groups:
        for group in sorted(list(groups), key=str):
            selected.append(groups[group].popleft())
            if not groups[group]:
                del groups[group]
            if len(selected) == 200:
                break
    records = []
    for row in selected:
        records.append(
            {
                "id": f"longbench-v2-{row['_id']}",
                "dataset": "longbench-v2",
                "subset": row.get("domain"),
                "type": "multiple_choice",
                "question": f"Context:\n{row['context']}\n\nQuestion:\n{row['question']}",
                "choices": {
                    "A": row["choice_A"],
                    "B": row["choice_B"],
                    "C": row["choice_C"],
                    "D": row["choice_D"],
                },
                "answer": row["answer"],
                "metadata": {
                    "sub_domain": row.get("sub_domain"),
                    "difficulty": row.get("difficulty"),
                    "length": row.get("length"),
                    "benchmark_category": "长上下文能力",
                    "benchmark_metric": "accuracy",
                    "recommended_max_tokens": 8192,
                    "regression_subset": True,
                },
            }
        )
    output = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records
    ).encode()
    data_path = PACKAGE / "longbench-v2.jsonl"
    data_path.write_bytes(output)
    manifest = {
        "name": "quanttrio-llmbench-data-longbench-v2",
        "version": "0.5.0",
        "package": "llmbench_data_longbench_v2",
        "source_revision": REVISION,
        "datasets": {
            "longbench-v2": {
                "file": "longbench-v2.jsonl",
                "count": len(records),
                "type": "multiple_choice",
                "category": "长上下文能力",
                "metric": "accuracy",
                "license": "Apache-2.0",
                "restriction": None,
                "source": SOURCE,
                "sha256": hashlib.sha256(output).hexdigest(),
                "recommended_max_tokens": 8192,
                "regression_subset": True,
            }
        },
    }
    (PACKAGE / "pack.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(records)} records ({len(output)} bytes) to {data_path}")


if __name__ == "__main__":
    main()
