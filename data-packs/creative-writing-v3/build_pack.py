from __future__ import annotations

import base64
import hashlib
import json
import urllib.request
from pathlib import Path

REVISION = "c7c3ceef54c40a8ae02dc1c2e1a5e40970fe5c0b"
ROOT = "https://api.github.com/repos/EQ-bench/creative-writing-bench/contents/data"
PACKAGE = Path(__file__).parent / "llmbench_data_creative_writing_v3"


def github_file(path: str) -> str:
    request = urllib.request.Request(
        f"{ROOT}/{path}?ref={REVISION}", headers={"Accept": "application/vnd.github+json"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return base64.b64decode(json.loads(response.read())["content"]).decode()


def main() -> None:
    prompts = json.loads(github_file("creative_writing_prompts_v3.json"))
    rubric = github_file("creative_writing_criteria.txt").strip()
    records = []
    for key, item in sorted(prompts.items(), key=lambda pair: int(pair[0])):
        modifier = (item.get("seed_modifiers") or [""])[0]
        prompt = item["writing_prompt"].replace("<SEED>", modifier)
        records.append(
            {
                "id": f"creative-writing-v3-{int(key):02d}",
                "dataset": "creative-writing-v3",
                "type": "judge",
                "question": prompt,
                "answer": None,
                "metadata": {
                    "category": item.get("category"),
                    "title": item.get("title"),
                    "rubric": rubric,
                    "benchmark_category": "写作和创作",
                    "benchmark_metric": "judge_score",
                    "capability": "chat",
                    "adapter": "judge",
                    "recommended_max_tokens": 4096,
                    "official_full_set": True,
                },
            }
        )
    output = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in records
    ).encode()
    (PACKAGE / "creative-writing-v3.jsonl").write_bytes(output)
    source = f"https://github.com/EQ-bench/creative-writing-bench/tree/{REVISION}/data"
    manifest = {
        "name": "quanttrio-llmbench-data-creative-writing-v3",
        "version": "0.5.0",
        "package": "llmbench_data_creative_writing_v3",
        "source_revision": REVISION,
        "datasets": {
            "creative-writing-v3": {
                "file": "creative-writing-v3.jsonl",
                "count": 32,
                "type": "judge",
                "category": "写作和创作",
                "metric": "judge_score",
                "license": "LicenseRef-Upstream-Unspecified",
                "restriction": "local build only",
                "source": source,
                "sha256": hashlib.sha256(output).hexdigest(),
                "recommended_max_tokens": 4096,
                "official_full_set": True,
            }
        },
    }
    (PACKAGE / "pack.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(records)} local-only prompts")


if __name__ == "__main__":
    main()
