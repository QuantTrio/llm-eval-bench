from __future__ import annotations

import gzip
import hashlib
import json
import urllib.request
from pathlib import Path

REVISION = "6d43fb980f9fee3c892a914eda09951f772ad10d"
SOURCE = f"https://raw.githubusercontent.com/openai/human-eval/{REVISION}/data/HumanEval.jsonl.gz"
PACKAGE = Path(__file__).parent / "llmbench_data_humaneval"


def main() -> None:
    with urllib.request.urlopen(SOURCE, timeout=60) as response:
        source = gzip.decompress(response.read()).decode("utf-8")
    records = []
    for line in source.splitlines()[:100]:
        item = json.loads(line)
        records.append(
            {
                "id": item["task_id"],
                "dataset": "humaneval",
                "type": "code",
                "question": item["prompt"],
                "answer": item["canonical_solution"],
                "metadata": {
                    "entry_point": item["entry_point"],
                    "test": item["test"],
                    "benchmark_category": "代码能力",
                    "benchmark_metric": "pass_at_1",
                    "capability": "agent",
                    "adapter": "remote_executor",
                    "recommended_max_tokens": 4096,
                    "regression_subset": True,
                },
            }
        )
    output = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records
    ).encode()
    data_path = PACKAGE / "humaneval.jsonl"
    data_path.write_bytes(output)
    manifest = {
        "name": "quanttrio-llmbench-data-humaneval",
        "version": "0.5.0",
        "package": "llmbench_data_humaneval",
        "source_revision": REVISION,
        "datasets": {
            "humaneval": {
                "file": "humaneval.jsonl",
                "count": len(records),
                "type": "code",
                "category": "代码能力",
                "metric": "pass_at_1",
                "license": "MIT",
                "restriction": None,
                "source": SOURCE,
                "sha256": hashlib.sha256(output).hexdigest(),
                "recommended_max_tokens": 4096,
                "regression_subset": True,
            }
        },
    }
    (PACKAGE / "pack.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(records)} records to {data_path}")


if __name__ == "__main__":
    main()
