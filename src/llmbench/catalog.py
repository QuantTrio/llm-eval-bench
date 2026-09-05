"""What benchmarks exist, what category they belong to, and whether we can run them.

`datasets.py` answers a different question — how to read the records. Sizes, metrics and
adapters are facts owned by `data/manifest.json` and each data pack's `pack.json`; this
module never restates them, it only reads them back.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from importlib.resources import files
from typing import Any

from .data_packs import external_dataset_resources
from .datasets import read_manifest

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


@dataclass(frozen=True, slots=True)
class BenchmarkCapability:
    """One stable category, the benchmark representing it, and what it takes to run it.

    `capability` is the target a user must configure before installing anything, which is
    why it lives here. Record counts and runtime adapters are read from the installed
    data instead, so they cannot drift out of step with it.
    """

    category: str
    benchmark: str
    dataset_id: str
    capability: str
    data_pack: str | None = None


BENCHMARK_CAPABILITIES = (
    BenchmarkCapability("AI Agent - 信息收集", "BrowseComp", "browsecomp", "agent", "browsecomp"),
    BenchmarkCapability(
        "AI Agent - 工具使用", "Terminal-Bench 2.0", "terminal-bench-2", "agent", "terminal-bench-2"
    ),
    BenchmarkCapability(
        "Agent能力评测", "Aider Polyglot", "aider-polyglot", "agent", "aider-polyglot"
    ),
    BenchmarkCapability(
        "OpenClaw智能体能力综合测评", "PinchBench", "pinchbench", "agent", "pinchbench"
    ),
    BenchmarkCapability("代码能力", "HumanEval", "humaneval", "agent", "humaneval"),
    BenchmarkCapability(
        "写作和创作", "Creative Writing v3", "creative-writing-v3", "chat", "creative-writing-v3"
    ),
    BenchmarkCapability(
        "图像向量嵌入", "MMEB-v2 Image", "mmeb-v2-image", "embedding", "mmeb-v2-image"
    ),
    BenchmarkCapability("多模态理解", "MMMU", "mmmu", "multimodal"),
    BenchmarkCapability("常识推理", "SimpleBench", "simple-bench", "chat", "simple-bench"),
    BenchmarkCapability("常识问答", "SimpleQA", "simpleqa", "chat", "simpleqa"),
    BenchmarkCapability("指令跟随", "IFBench", "ifbench", "chat", "ifbench"),
    BenchmarkCapability("数学推理", "AIME 2025", "aime-2025", "chat", "aime-2025"),
    BenchmarkCapability("生产力知识", "GDPval gold", "gdpval-gold", "agent", "gdpval-gold"),
    BenchmarkCapability("真实性评估", "TruthfulQA", "truthfulqa", "chat"),
    BenchmarkCapability("科学与综合推理", "GPQA Diamond", "gpqa-diamond", "chat"),
    BenchmarkCapability("综合能力", "AGIEval", "agieval", "chat", "agieval"),
    BenchmarkCapability(
        "编程与软件工程", "LiveCodeBench", "livecodebench", "agent", "livecodebench"
    ),
    BenchmarkCapability("自然语言理解", "SuperGLUE", "superglue", "chat", "superglue"),
    BenchmarkCapability("长上下文能力", "LongBench v2", "longbench-v2", "chat", "longbench-v2"),
    BenchmarkCapability("阅读理解", "DROP", "drop", "chat"),
)


def installed_datasets() -> list[dict[str, Any]]:
    """Every dataset readable right now: bundled in the wheel plus installed data packs."""
    datasets = {name: dict(metadata) for name, metadata in read_manifest().items()}
    for name, resource in external_dataset_resources(exclude_names=datasets).items():
        datasets[name] = {
            **resource.metadata,
            "pack": resource.pack,
            "pack_version": resource.pack_version,
        }
    return [dict(name=name, **metadata) for name, metadata in sorted(datasets.items())]


def capability_matrix(installed: set[str] | None = None) -> list[dict[str, Any]]:
    """The stable category matrix, annotated with what is installed and how big it is."""
    available = {item["name"]: item for item in installed_datasets()}
    names = available.keys() if installed is None else installed
    rows = []
    for item in BENCHMARK_CAPABILITIES:
        row = asdict(item)
        row["installed"] = item.dataset_id in names
        row["count"] = (available.get(item.dataset_id) or {}).get("count")
        rows.append(row)
    return rows


def benchmark_catalog() -> dict[str, Any]:
    resource = files("llmbench").joinpath("data", "benchmark_catalog.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def list_benchmarks(
    *, category: str | None = None, bundled_only: bool = False
) -> list[dict[str, Any]]:
    """The snapshotted DataLearner reference catalog, ranked by published reports."""
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
