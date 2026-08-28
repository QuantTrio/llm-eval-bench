#!/usr/bin/env python3
"""Build the committed, offline dataset bundle from pinned upstream revisions.

This is a maintainer-only script. Runtime installation never downloads data.
Run with the optional ``data`` dependencies installed.
"""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import io
import json
import random
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "src" / "llmbench" / "data"

MMLU_PRO_REV = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
MMLU_REDUX_REV = "4563cfa79b659d76716e5b6019f46bba85fa02db"
GPQA_REV = "56686c06f5e19865c153de0fdb11be3890014df7"
GSM8K_REV = "3101c7d5072418e28b9008a6636bde82a006892c"
CEVAL_REV = "617524a00b307ff6f9933702f724131fe12ca7ce"
HELLASWAG_REV = "218ec52e09a7e7462a5400043bb9a69a41d06b76"
TRUTHFULQA_REV = "d71c110897f5d31c5d7f309e7bc316c152f6f031"
CEVAL_SUBJECTS = """
accountant advanced_mathematics art_studies basic_medicine business_administration
chinese_language_and_literature civil_servant clinical_medicine college_chemistry
college_economics college_physics college_programming computer_architecture computer_network
discrete_mathematics education_science electrical_engineer
environmental_impact_assessment_engineer fire_engineer high_school_biology
high_school_chemistry high_school_chinese high_school_geography high_school_history
high_school_mathematics high_school_physics high_school_politics
ideological_and_moral_cultivation law legal_professional logic mao_zedong_thought marxism
metrology_engineer middle_school_biology middle_school_chemistry middle_school_geography
middle_school_history middle_school_mathematics middle_school_physics middle_school_politics
modern_chinese_history operating_system physician plant_protection probability_and_statistics
professional_tour_guide sports_science tax_accountant teacher_qualification
urban_and_rural_planner veterinary_medicine
""".split()


def download(url: str, destination: Path) -> Path:
    if destination.exists():
        return destination
    request = urllib.request.Request(url, headers={"User-Agent": "quanttrio-llmbench-data/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as out:
        shutil.copyfileobj(response, out)
    return destination


def record(
    *,
    item_id: str,
    dataset: str,
    kind: str,
    question: str,
    answer: str | None,
    choices: list[str] | None = None,
    subset: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "id": item_id,
        "dataset": dataset,
        "type": kind,
        "question": question,
        "answer": answer,
    }
    if choices:
        value["choices"] = {chr(65 + index): text for index, text in enumerate(choices)}
    if subset:
        value["subset"] = subset
    if metadata:
        value["metadata"] = metadata
    return value


def write_jsonl(name: str, rows: list[dict[str, Any]]) -> str:
    path = DATA_DIR / f"{name}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_mmlu_pro(cache: Path, limit: int) -> tuple[list[dict[str, Any]], str]:
    url = (
        "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/"
        f"{MMLU_PRO_REV}/data/test-00000-of-00001.parquet"
    )
    path = download(url, cache / "mmlu-pro-test.parquet")
    rows = pq.read_table(path).to_pylist()
    rows.sort(key=lambda row: int(row["question_id"]))
    output = [
        record(
            item_id=f"mmlu-pro-{row['question_id']}",
            dataset="mmlu-pro",
            kind="multiple_choice",
            question=row["question"],
            choices=row["options"],
            answer=chr(65 + int(row["answer_index"])),
            subset=row["category"],
            metadata={"source": row.get("src"), "benchmark_metric": "accuracy"},
        )
        for row in rows[:limit]
    ]
    return output, url


def build_mmlu_redux(cache: Path, limit: int) -> tuple[list[dict[str, Any]], str]:
    url = f"https://github.com/aryopg/mmlu-redux/archive/{MMLU_REDUX_REV}.zip"
    archive = download(url, cache / "mmlu-redux.zip")
    output: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive) as bundle:
        names = sorted(
            name for name in bundle.namelist() if "/mmlu_redux/" in name and name.endswith(".csv")
        )
        readers: list[tuple[str, list[dict[str, str]]]] = []
        for name in names:
            subset = Path(name).stem.removeprefix("mmlu_")
            raw = bundle.read(name)
            try:
                decoded = raw.decode("utf-8-sig")
            except UnicodeDecodeError:
                decoded = raw.decode("cp1252")
            header = decoded.splitlines()[0]
            delimiter = ";" if header.count(";") > header.count(",") else ","
            readers.append(
                (subset, list(csv.DictReader(io.StringIO(decoded), delimiter=delimiter)))
            )
        index = 0
        while len(output) < limit and any(rows for _, rows in readers):
            for subset, rows in readers:
                if index >= len(rows) or len(output) >= limit:
                    continue
                row = rows[index]
                choices = list(ast.literal_eval(row["choices"]))
                corrected = (row.get("correct_answer") or row["answer"]).strip()
                try:
                    answer_index = int(corrected)
                except ValueError:
                    if corrected in choices:
                        answer_index = choices.index(corrected)
                    elif len(corrected) == 1 and corrected.upper() in "ABCDEFGHIJ":
                        answer_index = ord(corrected.upper()) - 65
                    else:
                        continue
                if answer_index < 0 or answer_index >= len(choices):
                    continue
                output.append(
                    record(
                        item_id=f"mmlu-redux-{subset}-{index:04d}",
                        dataset="mmlu-redux",
                        kind="multiple_choice",
                        question=row["question"],
                        choices=choices,
                        answer=chr(65 + answer_index),
                        subset=subset,
                        metadata={
                            "error_type": row.get("error_type"),
                            "source": row.get("source"),
                            "benchmark_metric": "accuracy",
                        },
                    )
                )
            index += 1
    return output, url


def build_gpqa(cache: Path) -> tuple[list[dict[str, Any]], str]:
    url = f"https://raw.githubusercontent.com/idavidrein/gpqa/{GPQA_REV}/dataset.zip"
    archive = download(url, cache / "gpqa.zip")
    output: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive) as bundle:
        encrypted = bundle.read("dataset/gpqa_diamond.csv", pwd=b"deserted-untie-orchid")
        rows = csv.DictReader(io.StringIO(encrypted.decode("utf-8")))
        for index, row in enumerate(rows):
            if index >= 100:
                break
            answer_pairs = [
                (row["Correct Answer"], True),
                (row["Incorrect Answer 1"], False),
                (row["Incorrect Answer 2"], False),
                (row["Incorrect Answer 3"], False),
            ]
            rng = random.Random(f"gpqa-diamond-{row['Record ID']}")
            rng.shuffle(answer_pairs)
            correct_index = next(i for i, (_, correct) in enumerate(answer_pairs) if correct)
            output.append(
                record(
                    item_id=f"gpqa-diamond-{row['Record ID']}",
                    dataset="gpqa-diamond",
                    kind="multiple_choice",
                    question=row["Question"],
                    choices=[value for value, _ in answer_pairs],
                    answer=chr(65 + correct_index),
                    subset=row["High-level domain"],
                    metadata={"subdomain": row["Subdomain"], "benchmark_metric": "accuracy"},
                )
            )
    return output, url


def build_gsm8k(cache: Path, limit: int) -> tuple[list[dict[str, Any]], str]:
    url = (
        "https://raw.githubusercontent.com/openai/grade-school-math/"
        f"{GSM8K_REV}/grade_school_math/data/test.jsonl"
    )
    path = download(url, cache / "gsm8k-test.jsonl")
    output = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()[:limit]):
        row = json.loads(line)
        output.append(
            record(
                item_id=f"gsm8k-{index:04d}",
                dataset="gsm8k",
                kind="math",
                question=row["question"],
                answer=row["answer"],
                metadata={"benchmark_metric": "exact_match"},
            )
        )
    return output, url


def build_ceval(cache: Path, limit: int) -> tuple[list[dict[str, Any]], str]:
    base = f"https://huggingface.co/datasets/ceval/ceval-exam/resolve/{CEVAL_REV}"
    tables: list[tuple[str, list[dict[str, Any]]]] = []
    for subject in CEVAL_SUBJECTS:
        path = download(
            f"{base}/{subject}/val-00000-of-00001.parquet",
            cache / f"ceval-{subject}-val.parquet",
        )
        tables.append((subject, pq.read_table(path).to_pylist()))
    output: list[dict[str, Any]] = []
    index = 0
    while len(output) < limit and any(rows for _, rows in tables):
        for subject, rows in tables:
            if index >= len(rows) or len(output) >= limit:
                continue
            row = rows[index]
            output.append(
                record(
                    item_id=f"ceval-{subject}-{row['id']}",
                    dataset="ceval",
                    kind="multiple_choice",
                    question=row["question"],
                    choices=[row["A"], row["B"], row["C"], row["D"]],
                    answer=row["answer"],
                    subset=subject,
                    metadata={"benchmark_metric": "accuracy", "split": "val"},
                )
            )
        index += 1
    return output, base


def build_hellaswag(cache: Path, limit: int) -> tuple[list[dict[str, Any]], str]:
    url = (
        "https://huggingface.co/datasets/Rowan/hellaswag/resolve/"
        f"{HELLASWAG_REV}/data/validation-00000-of-00001.parquet"
    )
    path = download(url, cache / "hellaswag-validation.parquet")
    rows = pq.read_table(path).to_pylist()
    output = [
        record(
            item_id=f"hellaswag-{row['ind']}",
            dataset="hellaswag",
            kind="multiple_choice",
            question=f"Complete the following scenario:\n{row['ctx']}",
            choices=row["endings"],
            answer=chr(65 + int(row["label"])),
            subset=row["activity_label"],
            metadata={"source_id": row["source_id"], "benchmark_metric": "accuracy"},
        )
        for row in rows[:limit]
    ]
    return output, url


def build_truthfulqa(cache: Path, limit: int = 200) -> tuple[list[dict[str, Any]], str]:
    url = f"https://raw.githubusercontent.com/sylinrl/TruthfulQA/{TRUTHFULQA_REV}/data/mc_task.json"
    path = download(url, cache / "truthfulqa-mc.json")
    rows = json.loads(path.read_text(encoding="utf-8"))
    output: list[dict[str, Any]] = []
    for index, row in enumerate(rows[:limit]):
        pairs = list(row["mc1_targets"].items())
        choices = [text for text, _ in pairs]
        correct_index = next(position for position, (_, label) in enumerate(pairs) if label == 1)
        output.append(
            record(
                item_id=f"truthfulqa-{index:04d}",
                dataset="truthfulqa",
                kind="multiple_choice",
                question=row["question"],
                choices=choices,
                answer=chr(65 + correct_index),
                metadata={"benchmark_metric": "mc1_accuracy"},
            )
        )
    return output, url


def build_drop(cache: Path, limit: int) -> tuple[list[dict[str, Any]], str]:
    url = "https://openaipublic.blob.core.windows.net/simple-evals/drop_v0_dev.jsonl.gz"
    path = download(url, cache / "drop-dev.jsonl.gz")
    output: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= limit:
                break
            row = json.loads(line)
            output.append(
                record(
                    item_id=f"drop-{index:04d}",
                    dataset="drop",
                    kind="f1",
                    question=row["context"],
                    answer=row["ref_text"],
                    metadata={"benchmark_metric": "token_f1"},
                )
            )
    return output, url


def build_stress() -> tuple[list[dict[str, Any]], str]:
    prompts = [
        "Explain why deterministic evaluation settings matter for quantization regression.",
        "Summarize the trade-off between model latency and throughput in three bullets.",
        "Write a short Python function that computes a percentile from sorted values.",
        "Give a concise explanation of mixture-of-experts routing.",
        "用三句话解释模型量化为什么可能影响准确率。",
        "列出并发压测中需要单独统计的四类错误。",
        "Compare exact match and semantic similarity metrics.",
        "Describe one way to detect truncated model responses.",
        "What is time to first token and why does it matter?",
        "Explain how a fixed random seed improves benchmark reproducibility.",
    ]
    return [
        record(
            item_id=f"stress-{index:03d}",
            dataset="stress",
            kind="stress",
            question=prompt,
            answer=None,
            metadata={"synthetic": True},
        )
        for index, prompt in enumerate(prompts)
    ], "generated by QuantTrio"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    temporary = None
    cache = args.cache_dir
    if cache is None:
        temporary = tempfile.TemporaryDirectory(prefix="llmbench-data-")
        cache = Path(temporary.name)
    cache.mkdir(parents=True, exist_ok=True)

    builders = {
        "mmlu-pro": lambda: build_mmlu_pro(cache, args.limit),
        "mmlu-redux": lambda: build_mmlu_redux(cache, args.limit),
        "gpqa-diamond": lambda: build_gpqa(cache),
        "gsm8k": lambda: build_gsm8k(cache, args.limit),
        "ceval": lambda: build_ceval(cache, args.limit),
        "hellaswag": lambda: build_hellaswag(cache, args.limit),
        "truthfulqa": lambda: build_truthfulqa(cache, 200),
        "drop": lambda: build_drop(cache, args.limit),
        "stress": build_stress,
    }
    license_map = {
        "mmlu-pro": ("Apache-2.0", None, "multiple_choice", "Comprehensive"),
        "mmlu-redux": ("CC-BY-4.0", None, "multiple_choice", "Comprehensive"),
        "gpqa-diamond": ("CC-BY-4.0", None, "multiple_choice", "Science & Reasoning"),
        "gsm8k": ("MIT", None, "math", "Math Reasoning"),
        "ceval": ("CC-BY-NC-SA-4.0", "non-commercial use", "multiple_choice", "Comprehensive"),
        "hellaswag": ("MIT", None, "multiple_choice", "Common Sense Reasoning"),
        "truthfulqa": ("Apache-2.0", None, "multiple_choice", "Truthfulness"),
        "drop": ("Apache-2.0", None, "f1", "Reading Comprehension"),
        "stress": ("Apache-2.0", None, "stress", "Performance"),
    }
    manifest: dict[str, Any] = {}
    for name, builder in builders.items():
        rows, source = builder()
        checksum = write_jsonl(name, rows)
        license_name, restriction, kind, category = license_map[name]
        manifest[name] = {
            "file": f"{name}.jsonl",
            "count": len(rows),
            "type": kind,
            "category": category,
            "metric": (
                "none"
                if name == "stress"
                else "token_f1"
                if name == "drop"
                else "exact_match"
                if name == "gsm8k"
                else "accuracy"
            ),
            "license": license_name,
            "restriction": restriction,
            "source": source,
            "sha256": checksum,
        }
        print(f"{name}: {len(rows)} rows {checksum}")
    (DATA_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if temporary:
        temporary.cleanup()


if __name__ == "__main__":
    main()
