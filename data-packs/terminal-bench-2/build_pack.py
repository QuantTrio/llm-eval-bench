from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

REVISION = "2fd12b88aafdd04a52c298e3940bcb189f9766d6"
SOURCE = f"https://api.github.com/repos/harbor-framework/terminal-bench-2/git/trees/{REVISION}?recursive=1"
PACKAGE = Path(__file__).parent / "llmbench_data_terminal_bench_2"


def main() -> None:
    request = urllib.request.Request(SOURCE, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        tree = json.loads(response.read())["tree"]
    task_ids = sorted(
        item["path"].split("/", 1)[0]
        for item in tree
        if item["path"].count("/") == 1 and item["path"].endswith("/task.toml")
    )
    records = [
        {
            "id": f"terminal-bench-2-{task_id}",
            "dataset": "terminal-bench-2",
            "type": "agent",
            "question": f"Run official Terminal-Bench 2.0 task: {task_id}",
            "answer": None,
            "metadata": {
                "task_id": task_id,
                "benchmark_category": "AI Agent - 工具使用",
                "benchmark_metric": "success_rate",
                "capability": "agent",
                "adapter": "official_harness",
                "executor_image": f"quanttrio/llmbench-terminal-bench-2:{REVISION[:12]}",
                "executor_command": ["run-task", task_id],
                "network": False,
                "official_full_set": True,
            },
        }
        for task_id in task_ids
    ]
    output = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records
    ).encode()
    data_path = PACKAGE / "terminal-bench-2.jsonl"
    data_path.write_bytes(output)
    manifest = {
        "name": "quanttrio-llmbench-data-terminal-bench-2",
        "version": "0.5.0",
        "package": "llmbench_data_terminal_bench_2",
        "source_revision": REVISION,
        "datasets": {
            "terminal-bench-2": {
                "file": "terminal-bench-2.jsonl",
                "count": len(records),
                "type": "agent",
                "category": "AI Agent - 工具使用",
                "metric": "success_rate",
                "license": "Apache-2.0",
                "restriction": "requires version-matched remote executor image",
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
    print(f"Wrote {len(records)} task descriptors to {data_path}")


if __name__ == "__main__":
    main()
