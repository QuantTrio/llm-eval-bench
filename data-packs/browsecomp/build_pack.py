from __future__ import annotations

import csv
import hashlib
import io
import json
import urllib.request
from pathlib import Path

REVISION = "652c89d0ca9df547706735883097e9537d40dc47"
SOURCE = "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"
PACKAGE = Path(__file__).parent / "llmbench_data_browsecomp"


def main() -> None:
    with urllib.request.urlopen(SOURCE, timeout=60) as response:
        text = response.read().decode("utf-8")
    records = []
    for index, item in enumerate(csv.DictReader(io.StringIO(text))):
        if index >= 100:
            break
        records.append(
            {
                "id": f"browsecomp-{index:04d}",
                "dataset": "browsecomp",
                "type": "agent",
                "question": "[encrypted BrowseComp task]",
                "answer": None,
                "metadata": {
                    "encrypted_problem": item["problem"],
                    "encrypted_answer": item["answer"],
                    "canary": item["canary"],
                    "benchmark_category": "AI Agent - 信息收集",
                    "benchmark_metric": "judge_accuracy",
                    "capability": "agent",
                    "adapter": "remote_browser",
                    "recommended_max_tokens": 8192,
                    "regression_subset": True,
                    "sensitive": True,
                },
            }
        )
    output = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records
    ).encode()
    data_path = PACKAGE / "browsecomp.jsonl"
    data_path.write_bytes(output)
    manifest = {
        "name": "quanttrio-llmbench-data-browsecomp",
        "version": "0.5.0",
        "package": "llmbench_data_browsecomp",
        "source_revision": REVISION,
        "datasets": {
            "browsecomp": {
                "file": "browsecomp.jsonl",
                "count": len(records),
                "type": "agent",
                "category": "AI Agent - 信息收集",
                "metric": "judge_accuracy",
                "license": "MIT",
                "restriction": "encrypted; do not reveal questions or answers",
                "source": SOURCE,
                "sha256": hashlib.sha256(output).hexdigest(),
                "recommended_max_tokens": 8192,
                "regression_subset": True,
                "sensitive": True,
            }
        },
    }
    (PACKAGE / "pack.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(records)} encrypted records to {data_path}")


if __name__ == "__main__":
    main()
