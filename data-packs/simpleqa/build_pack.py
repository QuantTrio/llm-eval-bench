from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import urllib.request
from pathlib import Path

REVISION = "652c89d0ca9df547706735883097e9537d40dc47"
SOURCE = "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"
PACKAGE = Path(__file__).parent / "llmbench_data_simpleqa"


def main() -> None:
    with urllib.request.urlopen(SOURCE, timeout=60) as response:
        text = response.read().decode("utf-8")
    records = []
    for index, item in enumerate(csv.DictReader(io.StringIO(text))):
        if index >= 500:
            break
        upstream = ast.literal_eval(item["metadata"])
        records.append(
            {
                "id": f"simpleqa-{index:04d}",
                "dataset": "simpleqa",
                "type": "judge",
                "question": item["problem"],
                "answer": item["answer"],
                "metadata": {
                    "topic": upstream.get("topic"),
                    "answer_type": upstream.get("answer_type"),
                    "urls": upstream.get("urls") or [],
                    "benchmark_category": "常识问答",
                    "benchmark_metric": "judge_accuracy",
                    "capability": "chat",
                    "adapter": "judge",
                    "rubric": (
                        "Score 1 only when the response is factually equivalent to the reference "
                        f"answer {item['answer']!r}; otherwise score 0."
                    ),
                    "recommended_max_tokens": 4096,
                    "regression_subset": True,
                },
            }
        )
    output = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records
    ).encode()
    data_path = PACKAGE / "simpleqa.jsonl"
    data_path.write_bytes(output)
    manifest = {
        "name": "quanttrio-llmbench-data-simpleqa",
        "version": "0.5.0",
        "package": "llmbench_data_simpleqa",
        "source_revision": REVISION,
        "datasets": {
            "simpleqa": {
                "file": "simpleqa.jsonl",
                "count": len(records),
                "type": "judge",
                "category": "常识问答",
                "metric": "judge_accuracy",
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
