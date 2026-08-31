from __future__ import annotations

import hashlib
import json
import tempfile
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

REVISION = "a7b26602f8321cf71132eeb57c7f4d163e9b5a50"
SOURCE = (
    "https://huggingface.co/datasets/test-time-compute/aime_2025/resolve/"
    f"{REVISION}/data/test-00000-of-00001.parquet"
)
PACKAGE = Path(__file__).parent / "llmbench_data_aime_2025"


def main() -> None:
    with urllib.request.urlopen(SOURCE, timeout=60) as response:
        payload = response.read()
    with tempfile.NamedTemporaryFile(suffix=".parquet") as handle:
        handle.write(payload)
        handle.flush()
        rows = pq.read_table(handle.name).to_pylist()
    records = []
    for index, row in enumerate(rows):
        metadata = row.get("metadata") or {}
        records.append(
            {
                "id": f"aime-2025-{index + 1:02d}",
                "dataset": "aime-2025",
                "type": "math",
                "question": row["question"],
                "answer": str(row["answer"]),
                "metadata": {
                    "problem_type": metadata.get("problem_type"),
                    "difficulty": metadata.get("difficulty", "competition"),
                    "benchmark_category": "数学推理",
                    "benchmark_metric": "exact_match",
                    "recommended_max_tokens": 8192,
                    "official_full_set": True,
                },
            }
        )
    output = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records
    ).encode()
    data_path = PACKAGE / "aime-2025.jsonl"
    data_path.write_bytes(output)
    manifest = {
        "name": "quanttrio-llmbench-data-aime-2025",
        "version": "0.5.0",
        "package": "llmbench_data_aime_2025",
        "source_revision": REVISION,
        "datasets": {
            "aime-2025": {
                "file": "aime-2025.jsonl",
                "count": len(records),
                "type": "math",
                "category": "数学推理",
                "metric": "exact_match",
                "license": "MIT",
                "restriction": None,
                "source": SOURCE,
                "sha256": hashlib.sha256(output).hexdigest(),
                "recommended_max_tokens": 8192,
                "official_full_set": True,
            }
        },
    }
    (PACKAGE / "pack.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(records)} records to {data_path}")


if __name__ == "__main__":
    main()
