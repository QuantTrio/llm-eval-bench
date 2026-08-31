from __future__ import annotations

import base64
import hashlib
import json
import re
import urllib.request
from pathlib import Path

REVISION = "fbc2e429085bdedad7d1a236d2bc9bc18c95f16e"
SOURCE = (
    "https://api.github.com/repos/simple-bench/SimpleBench/contents/"
    f"simple_bench_public.json?ref={REVISION}"
)
PACKAGE = Path(__file__).parent / "llmbench_data_simple_bench"


def main() -> None:
    request = urllib.request.Request(SOURCE, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        envelope = json.loads(response.read())
    source = json.loads(base64.b64decode(envelope["content"]))
    records = []
    for item in source["eval_data"]:
        parts = re.split(r"\n([A-F])\.\s+", item["prompt"])
        question = parts[0].strip()
        choices = {parts[index]: parts[index + 1].strip() for index in range(1, len(parts), 2)}
        records.append(
            {
                "id": f"simple-bench-{item['question_id']}",
                "dataset": "simple-bench",
                "type": "multiple_choice",
                "question": question,
                "choices": choices,
                "answer": item["answer"],
                "metadata": {
                    "benchmark_category": "常识推理",
                    "benchmark_metric": "accuracy",
                    "recommended_max_tokens": 4096,
                    "official_public_full_set": True,
                },
            }
        )
    output = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records
    ).encode()
    data_path = PACKAGE / "simple-bench.jsonl"
    data_path.write_bytes(output)
    manifest = {
        "name": "quanttrio-llmbench-data-simple-bench",
        "version": "0.5.0",
        "package": "llmbench_data_simple_bench",
        "source_revision": REVISION,
        "datasets": {
            "simple-bench": {
                "file": "simple-bench.jsonl",
                "count": len(records),
                "type": "multiple_choice",
                "category": "常识推理",
                "metric": "accuracy",
                "license": "MIT",
                "restriction": None,
                "source": SOURCE,
                "sha256": hashlib.sha256(output).hexdigest(),
                "recommended_max_tokens": 4096,
                "official_public_full_set": True,
            }
        },
    }
    (PACKAGE / "pack.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(records)} records to {data_path}")


if __name__ == "__main__":
    main()
