from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from collections import defaultdict, deque
from pathlib import Path

REVISION = "7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f"
SOURCE = (
    f"https://api.github.com/repos/Aider-AI/polyglot-benchmark/git/trees/{REVISION}?recursive=1"
)
PACKAGE = Path(__file__).parent / "llmbench_data_aider_polyglot"
PATTERN = re.compile(r"^([^/]+)/exercises/practice/([^/]+)/\.docs/instructions\.md$")


def main() -> None:
    request = urllib.request.Request(SOURCE, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        tree = json.loads(response.read())["tree"]
    groups = defaultdict(deque)
    for item in tree:
        match = PATTERN.match(item["path"])
        if match:
            groups[match.group(1)].append(match.group(2))
    selected = []
    while len(selected) < 200 and groups:
        for language in sorted(list(groups)):
            selected.append((language, groups[language].popleft()))
            if not groups[language]:
                del groups[language]
            if len(selected) == 200:
                break
    records = [
        {
            "id": f"aider-polyglot-{language}-{exercise}",
            "dataset": "aider-polyglot",
            "subset": language,
            "type": "agent",
            "question": f"Run Aider Polyglot exercise {language}/{exercise}",
            "answer": None,
            "metadata": {
                "language": language,
                "exercise": exercise,
                "benchmark_category": "Agent能力评测",
                "benchmark_metric": "pass_rate_1",
                "capability": "agent",
                "adapter": "official_harness",
                "executor_image": f"quanttrio/llmbench-aider-polyglot:{REVISION[:12]}",
                "executor_command": ["run-task", language, exercise],
                "network": False,
                "regression_subset": True,
            },
        }
        for language, exercise in selected
    ]
    output = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records
    ).encode()
    data_path = PACKAGE / "aider-polyglot.jsonl"
    data_path.write_bytes(output)
    manifest = {
        "name": "quanttrio-llmbench-data-aider-polyglot",
        "version": "0.5.0",
        "package": "llmbench_data_aider_polyglot",
        "source_revision": REVISION,
        "datasets": {
            "aider-polyglot": {
                "file": "aider-polyglot.jsonl",
                "count": len(records),
                "type": "agent",
                "category": "Agent能力评测",
                "metric": "pass_rate_1",
                "license": "LicenseRef-Upstream-Mixed",
                "restriction": "requires version-matched remote executor image",
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
    print(f"Wrote {len(records)} task descriptors to {data_path}")


if __name__ == "__main__":
    main()
