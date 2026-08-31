from __future__ import annotations

import hashlib
import json
import tempfile
import urllib.request
from collections import defaultdict, deque
from pathlib import Path

import pyarrow.parquet as pq

REVISION = "11e7900cdcac61bc4daf59e65feb238acda98fbf"
SOURCE = f"https://huggingface.co/datasets/openai/gdpval/resolve/{REVISION}/data/train-00000-of-00001.parquet"
PACKAGE = Path(__file__).parent / "llmbench_data_gdpval_gold"


def main() -> None:
    with urllib.request.urlopen(SOURCE, timeout=120) as response:
        payload = response.read()
    with tempfile.NamedTemporaryFile(suffix=".parquet") as handle:
        handle.write(payload)
        handle.flush()
        rows = pq.read_table(handle.name).to_pylist()
    groups = defaultdict(deque)
    for row in rows:
        groups[str(row["occupation"])].append(row)
    selected = []
    while len(selected) < 200 and groups:
        for occupation in sorted(list(groups)):
            selected.append(groups[occupation].popleft())
            if not groups[occupation]:
                del groups[occupation]
            if len(selected) == 200:
                break
    records = []
    for row in selected:
        records.append(
            {
                "id": f"gdpval-{row['task_id']}",
                "dataset": "gdpval-gold",
                "type": "agent",
                "question": row["prompt"],
                "answer": None,
                "metadata": {
                    "sector": row["sector"],
                    "occupation": row["occupation"],
                    "reference_files": row.get("reference_files") or [],
                    "reference_file_urls": row.get("reference_file_urls") or [],
                    "deliverable_files": row.get("deliverable_files") or [],
                    "rubric": row.get("rubric_pretty"),
                    "rubric_json": row.get("rubric_json"),
                    "benchmark_category": "生产力知识",
                    "benchmark_metric": "judge_score",
                    "capability": "agent",
                    "adapter": "artifact_judge",
                    "executor_image": f"quanttrio/llmbench-gdpval:{REVISION[:12]}",
                    "executor_command": ["run-task", str(row["task_id"])],
                    "network": False,
                    "recommended_max_tokens": 8192,
                    "regression_subset": True,
                },
            }
        )
    output = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in records
    ).encode()
    (PACKAGE / "gdpval-gold.jsonl").write_bytes(output)
    manifest = {
        "name": "quanttrio-llmbench-data-gdpval-gold",
        "version": "0.5.0",
        "package": "llmbench_data_gdpval_gold",
        "source_revision": REVISION,
        "datasets": {
            "gdpval-gold": {
                "file": "gdpval-gold.jsonl",
                "count": len(records),
                "type": "agent",
                "category": "生产力知识",
                "metric": "judge_score",
                "license": "LicenseRef-GDPval-Upstream",
                "restriction": "local build only",
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
    print(f"Wrote {len(records)} local-only descriptors")


if __name__ == "__main__":
    main()
