#!/usr/bin/env python3
"""Snapshot DataLearner's public benchmark catalog for selection transparency.

The snapshot is documentation metadata only. It is not used to download or
redistribute benchmark samples.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "src" / "llmbench" / "data" / "benchmark_catalog.json"
BASE_URL = "https://www.datalearner.com"


def flight_text(page: str) -> str:
    chunks: list[str] = []
    pattern = re.compile(r"<script>self\.__next_f\.push\((\[.*?\])\)</script>", re.S)
    for match in pattern.finditer(page):
        try:
            value = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if len(value) > 1 and isinstance(value[1], str):
            chunks.append(value[1])
    return "".join(chunks)


def initial_benchmarks(page: str) -> list[dict[str, Any]]:
    text = flight_text(page)
    marker = '"initialBenchmarks":'
    start = text.find(marker)
    if start < 0:
        raise RuntimeError("DataLearner page no longer exposes initialBenchmarks")
    value, _ = json.JSONDecoder().raw_decode(text, start + len(marker))
    if not isinstance(value, list):
        raise RuntimeError("DataLearner initialBenchmarks is not a list")
    return value


def report_count(page: str) -> int:
    text = flight_text(page)
    matches = re.findall(r'"totalCount":(\d+)', text)
    return int(matches[-1]) if matches else 0


async def snapshot(output: Path, concurrency: int) -> None:
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(
        base_url=BASE_URL,
        timeout=30,
        follow_redirects=True,
        headers={"User-Agent": "quanttrio-llmbench-catalog/0.1"},
        limits=limits,
    ) as client:
        index = await client.get("/benchmarks")
        index.raise_for_status()
        benchmarks = initial_benchmarks(index.text)
        semaphore = asyncio.Semaphore(concurrency)

        async def enrich(item: dict[str, Any]) -> dict[str, Any]:
            async with semaphore:
                response = await client.get(f"/benchmarks/{item['benchmarkCode']}")
                response.raise_for_status()
            return {
                "code": item["benchmarkCode"],
                "name": item["shortName"],
                "full_name": item.get("fullName"),
                "category": item.get("category") or "Uncategorized",
                "language": item.get("language"),
                "problem_count": item.get("problemCount") or 0,
                "metric": item.get("metrics"),
                "institution": item.get("institution"),
                "dataset_url": item.get("datasetLink"),
                "paper_url": item.get("paperLink"),
                "view_count": item.get("viewCount") or 0,
                "report_count": report_count(response.text),
            }

        rows = await asyncio.gather(*(enrich(item) for item in benchmarks))
    rows.sort(key=lambda row: (row["category"], -row["report_count"], row["name"]))
    payload = {
        "source": f"{BASE_URL}/benchmarks",
        "snapshot_at": datetime.now(timezone.utc).isoformat(),
        "count": len(rows),
        "benchmarks": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} benchmarks to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--concurrency", type=int, default=6)
    args = parser.parse_args()
    asyncio.run(snapshot(args.output, args.concurrency))


if __name__ == "__main__":
    main()
