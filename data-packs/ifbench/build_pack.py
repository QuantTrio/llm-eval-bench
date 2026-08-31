from __future__ import annotations

import hashlib
import json
import tempfile
import urllib.request
from collections import defaultdict, deque
from pathlib import Path

import pyarrow.parquet as pq

REVISION = "2e8a48de45ff3bf41242f927254ca81b59ca3ae2"
SOURCE = (
    "https://huggingface.co/datasets/allenai/IFBench_test/resolve/"
    f"{REVISION}/data/train-00000-of-00001.parquet"
)
PACKAGE = Path(__file__).parent / "llmbench_data_ifbench"


def main() -> None:
    with urllib.request.urlopen(SOURCE, timeout=60) as response:
        payload = response.read()
    with tempfile.NamedTemporaryFile(suffix=".parquet") as handle:
        handle.write(payload)
        handle.flush()
        rows = pq.read_table(handle.name).to_pylist()
    groups = defaultdict(deque)
    for row in rows:
        instruction_ids = row.get("instruction_id_list") or ["unknown"]
        groups[str(instruction_ids[0])].append(row)
    selected = []
    while len(selected) < 200 and groups:
        for name in sorted(list(groups)):
            selected.append(groups[name].popleft())
            if not groups[name]:
                del groups[name]
            if len(selected) == 200:
                break
    records = [
        {
            "id": f"ifbench-{row['key']}",
            "dataset": "ifbench",
            "type": "instruction",
            "question": row["prompt"],
            "answer": None,
            "metadata": {
                "instruction_id_list": row.get("instruction_id_list") or [],
                "kwargs": row.get("kwargs") or [],
                "benchmark_category": "指令跟随",
                "benchmark_metric": "prompt_loose_accuracy",
                "capability": "chat",
                "adapter": "official_verifier",
                "recommended_max_tokens": 4096,
                "regression_subset": True,
            },
        }
        for row in selected
    ]
    output = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records
    ).encode()
    data_path = PACKAGE / "ifbench.jsonl"
    data_path.write_bytes(output)
    manifest = {
        "name": "quanttrio-llmbench-data-ifbench",
        "version": "0.5.0",
        "package": "llmbench_data_ifbench",
        "source_revision": REVISION,
        "datasets": {
            "ifbench": {
                "file": "ifbench.jsonl",
                "count": len(records),
                "type": "instruction",
                "category": "指令跟随",
                "metric": "prompt_loose_accuracy",
                "license": "ODC-By-1.0",
                "restriction": "Ai2 Responsible Use and third-party output terms",
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
