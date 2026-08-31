from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .schemas import RequestResult


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunArtifactWriter:
    """Incrementally persist a run so interruption never loses completed requests."""

    def __init__(self, output_dir: Path, *, checkpoint_every: int = 1) -> None:
        if checkpoint_every < 1:
            raise ValueError("checkpoint_every must be at least 1")
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.raw_path = output_dir / "raw_results.jsonl"
        self.events_path = output_dir / "events.jsonl"
        self.state_path = output_dir / "run_state.json"
        self.manifest_path = output_dir / "run_manifest.json"
        self.checkpoint_every = checkpoint_every
        self._since_sync = 0

    def existing_results(self) -> list[RequestResult]:
        if not self.raw_path.exists():
            return []
        results: list[RequestResult] = []
        with self.raw_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    results.append(RequestResult.from_dict(json.loads(line)))
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    raise ValueError(
                        f"Invalid resumable result at {self.raw_path}:{line_number}: {exc}"
                    ) from exc
        return results

    def append_result(
        self,
        result: RequestResult,
        *,
        completed: int,
        total: int,
        elapsed_seconds: float,
    ) -> None:
        with self.raw_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
            handle.flush()
            self._since_sync += 1
            if self._since_sync >= self.checkpoint_every:
                os.fsync(handle.fileno())
                self._since_sync = 0
        self.write_state(
            {
                "status": "running",
                "completed": completed,
                "total": total,
                "elapsed_seconds": elapsed_seconds,
                "updated_at": utc_now(),
            }
        )

    def event(self, event: str, **fields: Any) -> None:
        payload = {"timestamp": utc_now(), "event": event, **fields}
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            handle.flush()

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        self._atomic_json(self.manifest_path, manifest)

    def load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            raise ValueError(f"Resume manifest not found: {self.manifest_path}")
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def write_state(self, state: dict[str, Any]) -> None:
        self._atomic_json(self.state_path, state)

    def load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {}
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    @staticmethod
    def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
