from __future__ import annotations

import re
from typing import Any

import httpx

METRIC_PATTERN = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>[-+0-9.eE]+)$"
)


def metrics_url(base_url: str) -> str:
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    return root + "/metrics"


def parse_prometheus(text: str, *, model: str | None = None) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in text.splitlines():
        match = METRIC_PATTERN.match(line.strip())
        if not match:
            continue
        labels = match.group("labels") or ""
        if model and "model_name=" in labels and f'model_name="{model}"' not in labels:
            continue
        name = match.group("name")
        if name.startswith("vllm:"):
            label_suffix = ""
            reason = re.search(r'finished_reason="([^"]+)"', labels)
            if reason:
                label_suffix = f":{reason.group(1)}"
            values[name + label_suffix] = values.get(name + label_suffix, 0.0) + float(
                match.group("value")
            )
    return values


def metric_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    gauges = {"vllm:num_requests_running", "vllm:num_requests_waiting"}
    return {
        key: after.get(key, 0.0) if key in gauges else after.get(key, 0.0) - before.get(key, 0.0)
        for key in sorted(set(before) | set(after))
    }


class PrometheusCollector:
    def __init__(self, base_url: str, *, timeout: float = 10.0) -> None:
        self.url = metrics_url(base_url)
        self.timeout = timeout

    async def snapshot(self, *, model: str | None = None) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(self.url)
            response.raise_for_status()
        return {"url": self.url, "metrics": parse_prometheus(response.text, model=model)}
