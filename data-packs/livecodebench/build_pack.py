from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
SOURCE = f"https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/{REVISION}/test6.jsonl"
PACKAGE = Path(__file__).parent / "llmbench_data_livecodebench"


def main() -> None:
    with urllib.request.urlopen(SOURCE, timeout=120) as response:
        rows = [json.loads(line) for line in response.read().decode().splitlines() if line]
    rows = sorted(
        rows, key=lambda row: (row.get("contest_date", ""), row["question_id"]), reverse=True
    )[:100]
    records = []
    for row in rows:
        records.append(
            {
                "id": f"livecodebench-{row['question_id']}",
                "dataset": "livecodebench",
                "type": "agent",
                "question": row["question_content"],
                "answer": None,
                "metadata": {
                    "platform": row.get("platform"),
                    "difficulty": row.get("difficulty"),
                    "starter_code": row.get("starter_code"),
                    "public_test_cases": row.get("public_test_cases"),
                    "private_test_cases": row.get("private_test_cases"),
                    "benchmark_category": "编程与软件工程",
                    "benchmark_metric": "pass_at_1",
                    "capability": "agent",
                    "adapter": "official_harness",
                    "executor_image": f"quanttrio/llmbench-livecodebench:{REVISION[:12]}",
                    "executor_command": ["run-task", str(row["question_id"])],
                    "network": False,
                    "recommended_max_tokens": 8192,
                    "regression_subset": True,
                },
            }
        )
    output = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in records
    ).encode()
    (PACKAGE / "livecodebench.jsonl").write_bytes(output)
    manifest = {
        "name": "quanttrio-llmbench-data-livecodebench",
        "version": "0.5.0",
        "package": "llmbench_data_livecodebench",
        "source_revision": REVISION,
        "datasets": {
            "livecodebench": {
                "file": "livecodebench.jsonl",
                "count": len(records),
                "type": "agent",
                "category": "编程与软件工程",
                "metric": "pass_at_1",
                "license": "LicenseRef-LiveCodeBench-CC",
                "restriction": "local build only; review upstream terms",
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
    print(f"Wrote {len(records)} local-only records")


if __name__ == "__main__":
    main()
