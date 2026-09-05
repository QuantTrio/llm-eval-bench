"""Shared machinery for the data-pack builders.

Every pack does the same four things: fetch an upstream snapshot, convert its rows into
llmbench records, write one JSONL file, and write the `pack.json` manifest describing it.
Only the fetch and the conversion differ, so only those live in each pack's script.

Record and manifest layout is byte-exact on purpose: `pack.json` carries a SHA256 of the
JSONL, and `llmbench data verify` checks it, so a reordered key is a broken pack.

Maintainer-only. Needs `.[data]` and network access; runtime users never import this.
"""

from __future__ import annotations

import base64
import csv
import gzip
import hashlib
import io
import json
import tempfile
import time
import urllib.request
from collections import defaultdict, deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

Row = dict[str, Any]
GITHUB_JSON = {"Accept": "application/vnd.github+json"}


@dataclass(frozen=True, slots=True)
class Pack:
    """Everything about a pack that is a fact rather than a procedure."""

    dataset: str
    package: str
    revision: str
    source: str
    type: str
    category: str
    metric: str
    license: str
    version: str = "0.5.0"
    restriction: str | None = None
    recommended_max_tokens: int = 4096
    limit: int | None = None
    # Flags recorded in both the manifest and every record: regression_subset and friends.
    flags: dict[str, Any] = field(default_factory=dict)
    # Record metadata constant across the pack: capability, adapter, executor_image, ...
    item_metadata: dict[str, Any] = field(default_factory=dict)
    # A few packs predate the per-record budget and carry it only in the manifest.
    max_tokens_in_record: bool = True

    @property
    def filename(self) -> str:
        return f"{self.dataset}.jsonl"

    def directory(self, script: str) -> Path:
        return Path(script).resolve().parent / self.package

    def record(
        self,
        *,
        id: str,
        question: str,
        answer: str | None = None,
        choices: dict[str, str] | None = None,
        subset: str | None = None,
        type: str | None = None,
        metadata: Row | None = None,
        late_metadata: Row | None = None,
    ) -> Row:
        """One llmbench record, with the fields every pack shares filled in.

        `metadata` is per-row and leads. `late_metadata` fills per-row fields that belong
        among the pack constants: declare them in `item_metadata` as `...` to fix their
        position, or omit them there to have them appended.
        """
        record: Row = {"id": id, "dataset": self.dataset}
        if subset is not None:
            record["subset"] = subset
        record["type"] = type or self.type
        record["question"] = question
        if choices is not None:
            record["choices"] = choices
        record["answer"] = answer
        constants = dict(self.item_metadata)
        for key, value in (late_metadata or {}).items():
            constants[key] = value
        unfilled = [key for key, value in constants.items() if value is ...]
        if unfilled:
            raise ValueError(f"{self.dataset} record {id} is missing metadata: {unfilled}")
        record["metadata"] = {
            **(metadata or {}),
            "benchmark_category": self.category,
            "benchmark_metric": self.metric,
            **constants,
            **(
                {"recommended_max_tokens": self.recommended_max_tokens}
                if self.max_tokens_in_record
                else {}
            ),
            **self.flags,
        }
        return record

    def manifest(self, records: list[Row], payload: bytes) -> Row:
        return {
            "name": f"quanttrio-llmbench-data-{self.dataset}",
            "version": self.version,
            "package": self.package,
            "source_revision": self.revision,
            "datasets": {
                self.dataset: {
                    "file": self.filename,
                    "count": len(records),
                    "type": self.type,
                    "category": self.category,
                    "metric": self.metric,
                    "license": self.license,
                    "restriction": self.restriction,
                    "source": self.source,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "recommended_max_tokens": self.recommended_max_tokens,
                    **self.flags,
                }
            },
        }


def serialize(records: list[Row]) -> bytes:
    return "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records
    ).encode()


def build(
    pack: Pack,
    convert: Callable[[Pack], list[Row]],
    *,
    script: str,
    noun: str = "records",
) -> None:
    """Write the pack's JSONL and manifest — the only two files a pack ships."""
    records = convert(pack)
    payload = serialize(records)
    directory = pack.directory(script)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / pack.filename).write_bytes(payload)
    (directory / "pack.json").write_text(
        json.dumps(pack.manifest(records, payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} {noun} to {directory / pack.filename}")


# --- fetching -------------------------------------------------------------------


def download(url: str, *, timeout: int = 60, headers: dict[str, str] | None = None) -> bytes:
    """Fetch one URL, retrying with exponential backoff; upstream hosts rate-limit."""
    error: Exception | None = None
    request = urllib.request.Request(url, headers=headers or {})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except OSError as exc:
            error = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"failed to download after retries: {url}: {error}")


def read_json(url: str, *, timeout: int = 60, headers: dict[str, str] | None = None) -> Any:
    return json.loads(download(url, timeout=timeout, headers=headers))


def read_jsonl(url: str, *, timeout: int = 60) -> list[Row]:
    text = download(url, timeout=timeout).decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def read_jsonl_gz(url: str, *, timeout: int = 60) -> list[Row]:
    text = gzip.decompress(download(url, timeout=timeout)).decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def read_csv(url: str, *, timeout: int = 60) -> list[Row]:
    text = download(url, timeout=timeout).decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def read_parquet(url: str, *, timeout: int = 60) -> list[Row]:
    import pyarrow.parquet as pq

    with tempfile.NamedTemporaryFile(suffix=".parquet") as handle:
        handle.write(download(url, timeout=timeout))
        handle.flush()
        return pq.read_table(handle.name).to_pylist()


def github_file(repo: str, path: str, ref: str, *, timeout: int = 60) -> str:
    """One file's text through the contents API, which returns it base64-wrapped."""
    envelope = read_json(
        f"https://api.github.com/repos/{repo}/contents/{path}?ref={ref}",
        timeout=timeout,
        headers=GITHUB_JSON,
    )
    return base64.b64decode(envelope["content"]).decode("utf-8")


def github_tree(repo: str, ref: str, *, timeout: int = 60) -> list[Row]:
    return read_json(
        f"https://api.github.com/repos/{repo}/git/trees/{ref}?recursive=1",
        timeout=timeout,
        headers=GITHUB_JSON,
    )["tree"]


# --- selection ------------------------------------------------------------------


def group_by(rows: Iterable[Row], key: Callable[[Row], Any]) -> dict[Any, deque]:
    groups: dict[Any, deque] = defaultdict(deque)
    for row in rows:
        groups[key(row)].append(row)
    return groups


def balanced(groups: dict[Any, deque], limit: int) -> list[tuple[Any, Any]]:
    """Take round-robin across subsets, so a packaged subset stays representative."""
    remaining = {name: items for name, items in groups.items() if items}
    selected: list[tuple[Any, Any]] = []
    while len(selected) < limit and remaining:
        for name in sorted(remaining, key=str):
            selected.append((name, remaining[name].popleft()))
            if not remaining[name]:
                del remaining[name]
            if len(selected) == limit:
                break
    return selected
