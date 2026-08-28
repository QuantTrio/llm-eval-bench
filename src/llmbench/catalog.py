from __future__ import annotations

import json
from importlib.resources import files
from typing import Any

BUNDLED_CODE_MAP = {
    "mmlu-pro": "mmlu-pro",
    "mmlu-redux": "mmlu-redux",
    "gpqa-diamond": "gpqa-diamond",
    "gsm8k": "gsm8k",
    "c-eval": "ceval",
    "hellaswag": "hellaswag",
    "truthfulqa": "truthfulqa",
    "drop": "drop",
}


def benchmark_catalog() -> dict[str, Any]:
    resource = files("llmbench").joinpath("data", "benchmark_catalog.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def list_benchmarks(
    *, category: str | None = None, bundled_only: bool = False
) -> list[dict[str, Any]]:
    rows = []
    for item in benchmark_catalog()["benchmarks"]:
        bundled_as = BUNDLED_CODE_MAP.get(item["code"])
        if category and category.casefold() not in item["category"].casefold():
            continue
        if bundled_only and not bundled_as:
            continue
        rows.append({**item, "bundled_as": bundled_as})
    return sorted(rows, key=lambda row: (-row["report_count"], row["name"]))


def report_count_for_dataset(name: str) -> int | None:
    reverse = {dataset: code for code, dataset in BUNDLED_CODE_MAP.items()}
    code = reverse.get(name)
    if code is None:
        return None
    item = next(
        (entry for entry in benchmark_catalog()["benchmarks"] if entry["code"] == code),
        None,
    )
    return None if item is None else int(item["report_count"])
