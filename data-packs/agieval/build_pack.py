from __future__ import annotations

import base64
import hashlib
import json
import re
import urllib.request
from collections import defaultdict, deque
from pathlib import Path

REVISION = "84ab72d94318290aad2e4ec820d535a95a1f7552"
FILES = (
    "aqua-rat",
    "gaokao-biology",
    "gaokao-chemistry",
    "gaokao-english",
    "gaokao-geography",
    "gaokao-history",
    "gaokao-mathqa",
    "gaokao-physics",
    "jec-qa-ca",
    "jec-qa-kd",
    "logiqa-en",
    "logiqa-zh",
    "lsat-ar",
    "lsat-lr",
    "lsat-rc",
    "sat-en-without-passage",
    "sat-en",
    "sat-math",
)
ROOT = "https://api.github.com/repos/ruixiangcui/AGIEval/contents/data/v1_1"
PACKAGE = Path(__file__).parent / "llmbench_data_agieval"


def main() -> None:
    groups = defaultdict(deque)
    for subset in FILES:
        request = urllib.request.Request(
            f"{ROOT}/{subset}.jsonl?ref={REVISION}",
            headers={"Accept": "application/vnd.github+json"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            envelope = json.loads(response.read())
            source = base64.b64decode(envelope["content"]).decode("utf-8")
            for line in source.splitlines():
                item = json.loads(line)
                if item.get("options") and item.get("label"):
                    groups[subset].append(item)
    selected = []
    while len(selected) < 200 and groups:
        for subset in sorted(list(groups)):
            selected.append((subset, groups[subset].popleft()))
            if not groups[subset]:
                del groups[subset]
            if len(selected) == 200:
                break
    records = []
    for index, (subset, item) in enumerate(selected):
        choices = {
            chr(65 + offset): re.sub(r"^\([A-Z]\)\s*", "", str(option))
            for offset, option in enumerate(item["options"])
        }
        passage = str(item.get("passage") or "").strip()
        question = (passage + "\n\n" if passage else "") + item["question"]
        records.append(
            {
                "id": f"agieval-{subset}-{index:04d}",
                "dataset": "agieval",
                "subset": subset,
                "type": "multiple_choice",
                "question": question,
                "choices": choices,
                "answer": item["label"],
                "metadata": {
                    "benchmark_category": "综合能力",
                    "benchmark_metric": "accuracy",
                    "recommended_max_tokens": 4096,
                    "regression_subset": True,
                },
            }
        )
    output = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records
    ).encode()
    data_path = PACKAGE / "agieval.jsonl"
    data_path.write_bytes(output)
    source = f"https://github.com/ruixiangcui/AGIEval/tree/{REVISION}/data/v1_1"
    manifest = {
        "name": "quanttrio-llmbench-data-agieval",
        "version": "0.5.0",
        "package": "llmbench_data_agieval",
        "source_revision": REVISION,
        "datasets": {
            "agieval": {
                "file": "agieval.jsonl",
                "count": len(records),
                "type": "multiple_choice",
                "category": "综合能力",
                "metric": "accuracy",
                "license": "MIT",
                "restriction": None,
                "source": source,
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
