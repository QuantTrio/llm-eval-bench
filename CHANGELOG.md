# Changelog

## Unreleased

### 2.0.0.dev1 — one wheel with optional image processing

- Make normal wheel installation the README default; show optional image dependencies
  in a comment. Publish the single-wheel preview through GitHub Releases.
- Skip only absent legacy maintainer-source payloads in clean-checkout builder tests;
  the bundled core text and image data remain fully verified.

- Use distribution name `llm-bench` with an `image` extra for Pillow. One wheel includes
  the unchanged 3,300 text questions, 10 stress prompts, 500 MMMU questions and 535 PNGs.
  Extras select dependencies only; image data is never downloaded as a separate package.
- Route image datasets through the primary `run` command for Chat, Responses, Messages,
  and Gemini with native image content blocks, image validation, and prompt/image hashes.
- Include missing MMMU choice text in prompts and distinguish its 24 open questions from
  476 choice questions. Add local MMMU-style open-answer normalization and scoring.
- Verify every image's declared SHA-256 and reject changed images on resume. Keep full
  image payloads out of saved metadata and retain only per-request image inputs in memory.
- Preserve separate PNG bytes without resizing. Existing text corpus hashes are unchanged;
  older optional data packs cannot shadow core bundled dataset IDs.
- Real DeepSeek visual protocol checks and clean single-wheel acceptance are documented in
  `docs/releases/2026-09-05-single-wheel-image.md`.

### 2.0.0.dev0 — local preview

- Verify DeepSeek V4 Flash thinking mode over Chat/Responses/Messages, including JSON/SSE,
  thinking-only truncation, and real SIGINT/SIGKILL resume. Add process-level resume tests;
  see `docs/releases/2026-09-05-thinking-resume.md`. No runtime change was required.

- Present three primary commands: `run`, `compare`, and `datasets`; preserve legacy
  entry points for existing workflows. Add quick/standard/full presets and run modes.
- Evaluate text through Chat Completions, Responses, Anthropic Messages, and Gemini
  GenerateContent, including streaming, using a shared HTTP client without provider SDKs.
- Add provider shortcuts and explicit protocol selection; preserve custom Base URLs and
  proxy prefixes. Unauthenticated local endpoints no longer require a dummy API key.
- Every new evaluation requires an explicit model ID through CLI or YAML; model listing
  is a separate diagnostic. Errors do not trigger automatic protocol switching.
- Persist protocol, prompt hashes, request IDs, reasoning/cache usage, and provider output.
  Treat unknown usage as unavailable, not measured zero; flag failed evaluations invalid.
- Reject overwriting a run directory and incompatible resume settings.
- Make the README and comparison reports model-neutral; move detailed instructions to docs.
- This development version is tested locally; provider account access and live hardware
  results are separate acceptance checks. See `docs/releases/2026-09-05-local-preview.md`.

### Breaking

- Merge `llmbench sweep` into `llmbench stress`. `--concurrency` now accepts a
  comma-separated list; passing more than one level runs them in sequence and writes the
  sweep report alongside the per-level run directories. `llmbench sweep` still exists but
  only prints the replacement command and exits `2`:
  `llmbench stress --concurrency 1,4,8,16,32,64 --warmup-requests 20 --requests 200`.

### Changed

- `llmbench coverage` reports each category's installed record count, read from the data
  itself rather than from a second copy kept in the source.
- `llmbench stress` now writes a `run_manifest.json` per concurrency level, so
  `llmbench validate` accepts stress and sweep output.

### Fixed

- Correct the capability matrix: it claimed 100 PinchBench and 96 Creative Writing v3
  records against 23 and 32 actually packaged, and listed `remote_executor` for four packs
  whose data declares `official_harness`. Record counts, adapters and token budgets are now
  read from `manifest.json` and each pack's `pack.json` instead of being restated.
- Correct the stable matrix size: it is 20 categories, not 21.

### Internal

- `cli.py` is a command surface only: no networking, concurrency or file writing. Run
  orchestration moved to `runner.py`, `suite.py` and a new `session.py`; option resolution
  across CLI, YAML and resume moved to a new `runspec.py`.
- Merge `capabilities.py` into `catalog.py`; `datasets.py` is now purely a record loader.
- Replace 18 near-duplicate `data-packs/*/build_pack.py` scripts with one shared
  `data-packs/packbuild.py` plus a declaration per pack, covered by a test that rebuilds
  every shipped pack byte-for-byte offline.

## 1.0.1 - 2026-08-31

- Remove the MTEB retrieval mini dataset and text-embedding retrieval category from the full wheel.
- Catch the Python 3.10-specific `asyncio.TimeoutError` in the executor timeout path while
  preserving the stable public `TimeoutError` contract.
- Add one v1.0.1 full wheel containing the framework, CLI, core assets, and consolidated public
  data using the standards-compliant cross-platform `py3-none-any` tag.

## 1.0.0 - 2026-08-31

- Freeze the public CLI, YAML schema v2, report schema v2, and regression exit-code contract.
- Declare the 20-category representative benchmark matrix stable; unsupported targets stay unscored.
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
- Add a 20-category stable representative benchmark matrix; HLE and Fiction.liveBench remain catalog-only.
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
