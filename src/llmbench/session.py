"""Durable bookkeeping shared by every run mode.

Each mode differs in how it produces results; none of them differ in how results are
persisted, how progress is reported, or how a run is finalised. That part lives here.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

from .artifacts import RunArtifactWriter, utc_now
from .metrics import summarize
from .report import write_run_artifacts
from .schemas import RequestResult


class ResumeMismatch(ValueError):
    """A resumed directory does not describe the run being started."""


class RunSession:
    """Owns the output directory for one run: manifest, results, events, reports."""

    def __init__(
        self,
        directory: Path,
        *,
        mode: str,
        checkpoint_every: int = 1,
        progress_interval: float = 5.0,
    ) -> None:
        self.directory = directory
        self.mode = mode
        self.progress_interval = progress_interval
        self.writer = RunArtifactWriter(directory, checkpoint_every=checkpoint_every)
        self.previous_elapsed = 0.0
        self._resumed = 0
        self._started = time.perf_counter()
        self._last_progress = 0.0
        self._counters = {"errors": 0, "truncated": 0, "output_tokens": 0}

    def open(self, manifest: dict[str, Any], *, resume: bool = False) -> list[RequestResult]:
        """Record the manifest and return results already completed in this directory."""
        existing: list[RequestResult] = []
        if resume:
            previous = self.writer.load_manifest()
            if previous.get("fingerprint") != manifest["fingerprint"]:
                raise ResumeMismatch(
                    "resume configuration does not match run_manifest.json; "
                    "start a new output directory"
                )
            state = self.writer.load_state()
            self.previous_elapsed = float(state.get("elapsed_seconds") or 0.0)
            existing = self.writer.existing_results()
            if len({result.key for result in existing}) != len(existing):
                raise ResumeMismatch("resume results contain duplicate request keys")
        else:
            if any(
                path.exists()
                for path in (
                    self.writer.manifest_path,
                    self.writer.raw_path,
                    self.writer.state_path,
                )
            ):
                raise ResumeMismatch(
                    "output directory already contains a run; use --resume or a new --output-dir"
                )
            self.writer.write_manifest(manifest)
        self._resumed = len(existing)
        self._counters = {
            "errors": sum(result.error is not None for result in existing),
            "truncated": sum(result.finish_reason == "length" for result in existing),
            "output_tokens": sum(result.output_tokens for result in existing),
        }
        self._started = time.perf_counter()
        self.writer.event(
            "run_resumed" if resume else "run_started",
            run_id=manifest["run_id"],
            completed=self._resumed,
            total=manifest.get("request_count"),
            api=manifest.get("config", {}).get("api", "chat"),
        )
        return existing

    def warn(self, message: str, event: str, **fields: Any) -> None:
        print(message, file=sys.stderr)
        self.writer.event(event, message=message, **fields)

    def on_result(self, result: RequestResult, completed: int, total: int) -> None:
        """Runner callback: persist one result and emit throttled progress."""
        self._counters["errors"] += result.error is not None
        self._counters["truncated"] += result.finish_reason == "length"
        self._counters["output_tokens"] += result.output_tokens
        now = time.perf_counter()
        self.writer.append_result(
            result,
            completed=completed,
            total=total,
            elapsed_seconds=self.previous_elapsed + now - self._started,
        )
        self.writer.event(
            "request_completed",
            dataset=result.dataset,
            question_id=result.question_id,
            sample_id=result.sample_id,
            completed=completed,
            total=total or None,
            error_type=result.error_type,
            truncated=result.finish_reason == "length",
            attempts=result.attempts,
        )
        final = bool(total) and completed == total
        if now - self._last_progress < self.progress_interval and not final:
            return
        self._last_progress = now
        elapsed = max(now - self._started, 1e-9)
        qps = max(completed - self._resumed, 0) / elapsed
        eta = (total - completed) / qps if total and qps > 0 else None
        payload = {
            "completed": completed,
            "total": total or None,
            "qps": round(qps, 3),
            "eta_seconds": None if eta is None else round(eta, 1),
            **self._counters,
        }
        if sys.stderr.isatty():
            end = "\n" if final else "\r"
            print(
                f"[{completed}/{total or '?'}] qps={qps:.2f} eta={payload['eta_seconds']}s "
                f"errors={self._counters['errors']} truncated={self._counters['truncated']}",
                end=end,
                file=sys.stderr,
            )
        else:
            print(json.dumps({"event": "progress", **payload}), file=sys.stderr)
        self.writer.event("progress", **payload)

    def close(
        self,
        results: list[RequestResult],
        *,
        elapsed: float,
        run_id: str,
        model: str,
        base_url: str,
        config: dict[str, Any],
        total: int | None = None,
        extra_summary: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Path]]:
        """Summarise, write every report, and mark the run completed."""
        elapsed += self.previous_elapsed
        summary = summarize(
            results,
            run_id=run_id,
            mode=self.mode,
            model=model,
            base_url=base_url,
            elapsed_seconds=elapsed,
            config=config,
        )
        summary.update(extra_summary or {})
        summary["evaluation"] = {
            "complete": total is None or len(results) == total,
            "expected_requests": total,
            "completed_requests": len(results),
            "protocol": config.get("api", "chat"),
            "preset": config.get("preset"),
            "scope": "selected_records",
        }
        if not summary["evaluation"]["complete"]:
            summary["quality"]["quality_valid"] = False
        paths = write_run_artifacts(self.directory, summary, results)
        self.writer.write_state(
            {
                "status": "completed",
                "completed": len(results),
                "total": len(results) if total is None else total,
                "elapsed_seconds": elapsed,
                "updated_at": utc_now(),
            }
        )
        self.writer.event("run_completed", run_id=run_id, completed=len(results))
        return summary, paths
