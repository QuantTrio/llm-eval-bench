# Changelog

## 1.0.1 - 2026-08-31

- Catch the Python 3.10-specific `asyncio.TimeoutError` in the executor timeout path while
  preserving the stable public `TimeoutError` contract.
- Add one consolidated v0.5.0 wheel containing all 13 publicly redistributable data packs,
  integrity metadata, source revisions, and license notices.
- Add one v1.0.1 full wheel containing the framework, CLI, core assets, and consolidated public
  data using the standards-compliant cross-platform `py3-none-any` tag.

## 1.0.0 - 2026-08-31

- Freeze the public CLI, YAML schema v2, report schema v2, and regression exit-code contract.
- Declare the 21-category representative benchmark matrix stable; unsupported targets stay unscored.
- Freeze data-wheel discovery and integrity metadata while retaining v0.5.0 wheel compatibility.
- Publish migration and stable-interface documentation for v0.1.1 through v1.0.0.
- Promote package metadata to Production/Stable after the full quality, resume, comparison, report,
  capability-routing, and executor contract test suite passed.

## 0.6.0 - 2026-08-31

- Add self-contained category radar, dataset bars, question drilldown, and sweep curves.
- Add paired baseline/candidate raw-output drilldown to quantization comparison reports.
- Add JSON Schema v2 validation for run summaries, manifests, and raw JSONL records.
- Use collision-resistant UTC-plus-UUID run IDs while retaining read compatibility for v1 reports.
- Test Python 3.10, 3.11, and 3.12 and publish build artifacts with CycloneDX SBOM metadata.
- Document migration from v0.1.x commands, metrics, artifacts, and configuration.

## 0.5.0 - 2026-08-31

- Add capability-specific suite routing for chat, multimodal, embedding, Judge, and executor tasks.
- Add entry-point discovery and integrity verification for optional benchmark data wheels.
- Add a 21-category stable representative benchmark matrix; HLE and Fiction.liveBench remain catalog-only.
- Add 18 reproducible data-wheel projects, including large MMMU and LongBench release assets.
- Preserve encrypted BrowseComp records and withhold sensitive plaintext from reports.
- Mark missing model capabilities as unsupported rather than scoring them as failures.

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
