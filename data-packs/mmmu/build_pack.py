from __future__ import annotations

import ast
import base64
import hashlib
import json
import mimetypes
import tempfile
import time
import urllib.request
from collections import defaultdict, deque
from pathlib import Path

import pyarrow.parquet as pq

REVISION = "98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68"
API = "https://huggingface.co/api/datasets/MMMU/MMMU"
ROOT = f"https://huggingface.co/datasets/MMMU/MMMU/resolve/{REVISION}"
PACKAGE = Path(__file__).parent / "llmbench_data_mmmu"


def download(url: str, timeout: int) -> bytes:
    error = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return response.read()
        except OSError as exc:
            error = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"failed to download after retries: {url}: {error}")


def main() -> None:
    siblings = json.loads(download(API, 60))["siblings"]
    paths = sorted(
        item["rfilename"]
        for item in siblings
        if item["rfilename"].endswith("/validation-00000-of-00001.parquet")
    )
    groups = defaultdict(deque)
    for path in paths:
        payload = download(f"{ROOT}/{path}", 180)
        with tempfile.NamedTemporaryFile(suffix=".parquet") as handle:
            handle.write(payload)
            handle.flush()
            rows = pq.read_table(handle.name).to_pylist()
        groups[path.split("/", 1)[0]].extend(rows)
    selected = []
    while len(selected) < 500 and groups:
        for subject in sorted(list(groups)):
            selected.append((subject, groups[subject].popleft()))
            if not groups[subject]:
                del groups[subject]
            if len(selected) == 500:
                break
    records = []
    for subject, row in selected:
        assets = []
        for index in range(1, 8):
            image = row.get(f"image_{index}")
            if not image or not image.get("bytes"):
                continue
            mime = mimetypes.guess_type(image.get("path") or "image.png")[0] or "image/png"
            assets.append(f"data:{mime};base64," + base64.b64encode(image["bytes"]).decode())
        options = ast.literal_eval(row["options"])
        records.append(
            {
                "id": f"mmmu-{row['id']}",
                "dataset": "mmmu",
                "subset": subject,
                "type": "multiple_choice",
                "question": row["question"],
                "choices": {chr(65 + i): str(value) for i, value in enumerate(options)},
                "answer": row["answer"],
                "metadata": {
                    "assets": assets,
                    "difficulty": row.get("topic_difficulty"),
                    "subfield": row.get("subfield"),
                    "benchmark_category": "多模态理解",
                    "benchmark_metric": "accuracy",
                    "capability": "multimodal",
                    "adapter": "multimodal_chat",
                    "recommended_max_tokens": 4096,
                    "regression_subset": True,
                },
            }
        )
    output = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in records
    ).encode()
    (PACKAGE / "mmmu.jsonl").write_bytes(output)
    manifest = {
        "name": "quanttrio-llmbench-data-mmmu",
        "version": "0.5.0",
        "package": "llmbench_data_mmmu",
        "source_revision": REVISION,
        "datasets": {
            "mmmu": {
                "file": "mmmu.jsonl",
                "count": len(records),
                "type": "multiple_choice",
                "category": "多模态理解",
                "metric": "accuracy",
                "license": "Apache-2.0",
                "restriction": None,
                "source": ROOT,
                "sha256": hashlib.sha256(output).hexdigest(),
                "recommended_max_tokens": 4096,
                "regression_subset": True,
            }
        },
    }
    (PACKAGE / "pack.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(records)} records ({len(output)} bytes)")


if __name__ == "__main__":
    main()
