from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.request
from collections import defaultdict, deque
from pathlib import Path

import pyarrow.parquet as pq

REVISION = "5a81a4c7271a2a2a312b9a690f0c2fde837e4c29"
SOURCE = (
    f"https://huggingface.co/datasets/cais/hle/resolve/{REVISION}/data/test-00000-of-00001.parquet"
)
PACKAGE = Path(__file__).parent / "llmbench_data_hle"


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if not token and token_path.exists():
        token = token_path.read_text(encoding="utf-8").strip()
    if not token:
        raise SystemExit(
            "HF_TOKEN or ~/.cache/huggingface/token is required after accepting cais/hle terms"
        )
    request = urllib.request.Request(SOURCE, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(request, timeout=300) as response:
        payload = response.read()
    with tempfile.NamedTemporaryFile(suffix=".parquet") as handle:
        handle.write(payload)
        handle.flush()
        rows = pq.read_table(handle.name).to_pylist()
    groups = defaultdict(deque)
    for row in rows:
        groups[str(row.get("subject") or row.get("category") or "unknown")].append(row)
    selected = []
    while len(selected) < 100 and groups:
        for subject in sorted(list(groups)):
            selected.append(groups[subject].popleft())
            if not groups[subject]:
                del groups[subject]
            if len(selected) == 100:
                break
    records = []
    for row in selected:
        image = row.get("image")
        records.append(
            {
                "id": f"hle-{row['id']}",
                "dataset": "hle",
                "type": "judge",
                "question": row["question"],
                "answer": str(row["answer"]),
                "metadata": {
                    "subject": row.get("subject"),
                    "assets": [image] if image else [],
                    "rubric": (
                        "Score 1 only when the final answer is equivalent to the reference answer."
                    ),
                    "benchmark_category": "综合评估",
                    "benchmark_metric": "judge_accuracy",
                    "capability": "multimodal",
                    "adapter": "multimodal_judge",
                    "recommended_max_tokens": 8192,
                    "regression_subset": True,
                },
            }
        )
    output = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in records
    ).encode()
    (PACKAGE / "hle.jsonl").write_bytes(output)
    manifest = {
        "name": "quanttrio-llmbench-data-hle",
        "version": "0.5.0",
        "package": "llmbench_data_hle",
        "source_revision": REVISION,
        "datasets": {
            "hle": {
                "file": "hle.jsonl",
                "count": len(records),
                "type": "judge",
                "category": "综合评估",
                "metric": "judge_accuracy",
                "license": "MIT",
                "restriction": "gated; local build only",
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
    print(f"Wrote {len(records)} gated local-only records")


if __name__ == "__main__":
    main()
