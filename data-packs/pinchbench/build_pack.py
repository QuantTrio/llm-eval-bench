from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from pathlib import Path

REVISION = "97118e1f6bb1e8993458f84ed85927f5e328f174"
SOURCE = f"https://api.github.com/repos/Studio-Intrinsic/clawbench/git/trees/{REVISION}?recursive=1"
PACKAGE = Path(__file__).parent / "llmbench_data_pinchbench"


def main() -> None:
    request = urllib.request.Request(SOURCE, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        tree = json.loads(response.read())["tree"]
    tasks = sorted(
        path
        for item in tree
        if (path := item["path"]).startswith("tasks/task_")
        and path.endswith(".md")
        and "TEMPLATE" not in path
    )
    records = []
    for path in tasks:
        task_id = re.sub(r"\.md$", "", path.split("/", 1)[1])
        records.append(
            {
                "id": f"pinchbench-{task_id}",
                "dataset": "pinchbench",
                "type": "agent",
                "question": f"Run official PinchBench task: {task_id}",
                "answer": None,
                "metadata": {
                    "task_id": task_id,
                    "benchmark_category": "OpenClaw智能体能力综合测评",
                    "benchmark_metric": "success_rate",
                    "capability": "agent",
                    "adapter": "official_harness",
                    "executor_image": f"quanttrio/llmbench-pinchbench:{REVISION[:12]}",
                    "executor_command": ["run-task", task_id],
                    "network": True,
                    "official_full_set": True,
                },
            }
        )
    output = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in records
    ).encode()
    (PACKAGE / "pinchbench.jsonl").write_bytes(output)
    manifest = {
        "name": "quanttrio-llmbench-data-pinchbench",
        "version": "0.5.0",
        "package": "llmbench_data_pinchbench",
        "source_revision": REVISION,
        "datasets": {
            "pinchbench": {
                "file": "pinchbench.jsonl",
                "count": len(records),
                "type": "agent",
                "category": "OpenClaw智能体能力综合测评",
                "metric": "success_rate",
                "license": "MIT",
                "restriction": "requires remote OpenClaw executor and task-specific accounts",
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
    print(f"Wrote {len(records)} task descriptors")


if __name__ == "__main__":
    main()
