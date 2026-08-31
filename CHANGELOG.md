# Changelog

## 0.4.0 - 2026-08-31

- Add warmup, fixed-RPS pacing, worker ramp, prompt profiles, and concurrency sweep reports.
- Add per-run Prometheus baselines and vLLM counter deltas.
- Separate user-observed latency from final-attempt latency.
- Add the remote executor API, client, SSE events, cancellation, and artifacts.
- Add rootless-container defaults, ephemeral-key redaction, limits, and allowlisted proxy egress.

## 0.3.0 - 2026-08-31

- Add schema-v2 YAML run configuration with explicit CLI precedence.
- Require comparison-compatible manifests and identical paired request keys.
- Add question-level transitions and deterministic paired bootstrap confidence intervals.
- Add YAML regression gates and stable CI exit codes.

## 0.2.0 - 2026-08-31

- Raise the default evaluation output budget to 4096 tokens.
- Add live progress events, incremental JSONL checkpoints, strict resume, and bounded workers.
- Add reproducibility manifests with dataset and question fingerprints.
- Separate native metrics and remove misleading cross-metric accuracy aliases.
- Add truncation validity and reasoning-block-aware answer parsing.

## 0.1.1 - 2026-08-28

- Make the public documentation and examples model-neutral.

## 0.1.0 - 2026-08-28

- Add automatic OpenAI-compatible model discovery.
- Add quality, stress, and quality-under-load run modes.
- Bundle nine offline datasets/resources with pinned provenance.
- Add choice, math, exact-match, and token-F1 scorers.
- Add streaming TTFT/TPOT, latency percentiles, throughput, retry, and error metrics.
- Add dataset/category/question-type reports and quantization regression comparison.
- Add a 157-entry DataLearner benchmark catalog snapshot.
