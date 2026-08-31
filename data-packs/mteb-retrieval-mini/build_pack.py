from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

REVISION = "6c1bcf74b13dfd823aff056b79d4d93e702f19c7"
ROOT = f"https://huggingface.co/datasets/mteb/arguana/resolve/{REVISION}"
PACKAGE = Path(__file__).parent / "llmbench_data_mteb_retrieval_mini"


def lines(path: str) -> list[dict]:
    with urllib.request.urlopen(f"{ROOT}/{path}", timeout=120) as response:
        return [json.loads(line) for line in response.read().decode().splitlines() if line]


def main() -> None:
    corpus = {
        row["_id"]: (row.get("title", "") + "\n" + row["text"]).strip()
        for row in lines("corpus.jsonl")
    }
    queries = {row["_id"]: row["text"] for row in lines("queries.jsonl")}
    positives = {}
    for row in lines("qrels/test.jsonl"):
        if int(row.get("score", 0)) > 0:
            positives.setdefault(row["query-id"], row["corpus-id"])
    corpus_ids = sorted(corpus)
    records = []
    valid_query_ids = [
        query_id
        for query_id in sorted(set(queries) & set(positives))
        if positives[query_id] in corpus
    ]
    for query_id in valid_query_ids[:500]:
        positive_id = positives[query_id]
        negative_id = next(value for value in corpus_ids if value != positive_id)
        records.append(
            {
                "id": f"mteb-arguana-{query_id}",
                "dataset": "mteb-retrieval-mini",
                "type": "embedding",
                "question": queries[query_id],
                "answer": corpus[positive_id],
                "metadata": {
                    "positive": corpus[positive_id],
                    "negatives": [corpus[negative_id]],
                    "benchmark_category": "文本向量检索",
                    "benchmark_metric": "recall_at_1",
                    "capability": "embedding",
                    "adapter": "embedding",
                    "recommended_max_tokens": 1,
                    "regression_subset": True,
                },
            }
        )
    output = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in records
    ).encode()
    (PACKAGE / "mteb-retrieval-mini.jsonl").write_bytes(output)
    source = f"https://huggingface.co/datasets/mteb/arguana/tree/{REVISION}"
    manifest = {
        "name": "quanttrio-llmbench-data-mteb-retrieval-mini",
        "version": "0.5.0",
        "package": "llmbench_data_mteb_retrieval_mini",
        "source_revision": REVISION,
        "datasets": {
            "mteb-retrieval-mini": {
                "file": "mteb-retrieval-mini.jsonl",
                "count": len(records),
                "type": "embedding",
                "category": "文本向量检索",
                "metric": "recall_at_1",
                "license": "CC-BY-SA-4.0",
                "restriction": None,
                "source": source,
                "sha256": hashlib.sha256(output).hexdigest(),
                "recommended_max_tokens": 1,
                "regression_subset": True,
            }
        },
    }
    (PACKAGE / "pack.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(records)} retrieval records")


if __name__ == "__main__":
    main()
