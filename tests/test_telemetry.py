from __future__ import annotations

from llmbench.telemetry import metric_delta, metrics_url, parse_prometheus


def test_prometheus_parser_filters_model_and_deltas_counters() -> None:
    before = parse_prometheus(
        "\n".join(
            [
                'vllm:num_requests_running{model_name="model-a"} 2',
                'vllm:num_requests_waiting{model_name="model-a"} 1',
                'vllm:generation_tokens_total{model_name="model-a"} 100',
                'vllm:request_success_total{finished_reason="stop",model_name="model-a"} 5',
                'vllm:generation_tokens_total{model_name="model-b"} 999',
            ]
        ),
        model="model-a",
    )
    after = parse_prometheus(
        "\n".join(
            [
                'vllm:num_requests_running{model_name="model-a"} 3',
                'vllm:num_requests_waiting{model_name="model-a"} 0',
                'vllm:generation_tokens_total{model_name="model-a"} 140',
                'vllm:request_success_total{finished_reason="stop",model_name="model-a"} 8',
            ]
        ),
        model="model-a",
    )
    delta = metric_delta(before, after)
    assert delta["vllm:num_requests_running"] == 3
    assert delta["vllm:num_requests_waiting"] == 0
    assert delta["vllm:generation_tokens_total"] == 40
    assert delta["vllm:request_success_total:stop"] == 3
    assert metrics_url("http://localhost:8000/v1") == "http://localhost:8000/metrics"
