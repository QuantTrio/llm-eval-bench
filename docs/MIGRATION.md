# Migration guide

## 2.0.0.dev0 to 2.0.0.dev1

The distribution is now `llm-bench`; the executable/import remain `llmbench`. Install in a
clean virtual environment when moving from a `quanttrio-llmbench` distribution to avoid
two distributions owning the same import files. No public index upload is implied.

One wheel contains all 3,300 text records, 10 stress prompts, 500 MMMU records and 535 PNGs.
`pip install './llm_bench-2.0.0.dev1-py3-none-any.whl[image]'` installs the same data plus the
optional Pillow dependency; it does not fetch a second dataset wheel. Plain installation
also contains the image bytes, but does not require Pillow for text evaluation.

Use `run --dataset mmmu --model VISION_MODEL_ID` after installing `[image]`. The new native
image path includes choices, corrects 24 open-question types, and fingerprints image contents.
Do not compare old MMMU results produced without choices to the corrected run. Open questions
use a local MMMU-style scorer with documented differences from the official evaluator.
Core bundled dataset IDs take precedence over legacy optional packs with the same ID.

See [the single-wheel acceptance record](releases/2026-09-05-single-wheel-image.md).

## 1.x to 2.0.0.dev0

This is a local development preview. `run`, `compare`, and `datasets` are the primary
commands; existing advanced commands remain callable but hidden from the main help.
Every new run requires a model ID using `--model` or configuration. Auto-selection of the
first advertised model is removed. Resume uses the recorded ID and rejects changed settings.
Use `--api chat|responses|messages|generate-content` for custom endpoints, or a provider
shortcut. Requests never switch protocol after a failure.

Use explicit datasets and limits for established evaluations; quick/standard/full presets
are convenient sample selections. Non-chat protocols omit unspecified temperature/top-p;
inspect the saved config. Protocol and prompt hashes now participate in comparability checks.
Re-run old evaluations with this version for a strict paired comparison. Artifact filenames
and schema version 2 are retained; new optional request fields provide usage and request IDs.
Unknown token usage is null in summaries; clients must allow unavailable token throughput.
The core package version is `2.0.0.dev0`; the historical full-data bundle remains a v1 artifact.

See [USAGE.md](USAGE.md) and [API_COMPATIBILITY.md](API_COMPATIBILITY.md).

New runs use schema version 2. Existing v0.1.x output remains readable for manual inspection, but
must not be mixed with a schema-v2 run in a paired regression comparison.

## v0.1.x to v0.2.0

- Quality commands now default to `max_tokens=4096`; stress runs retain 128.
- Runs append `raw_results.jsonl` and `events.jsonl` as requests finish.
- Resume an interrupted directory with `llmbench run --resume RUN_DIR`.
- DROP is reported as token F1, not accuracy; `pass@k` is emitted only for repeated binary tasks.
- A dataset with more than 5% truncated responses is marked `quality_valid=false`.

## v0.2.0 to v0.3.0

- Put reusable target and run settings in a `schema_version: 2` YAML file.
- Replace ad-hoc score subtraction with `llmbench compare --baseline DIR --candidate DIR`.
- CI exit codes are stable: 0 pass, 2 regression, 3 incomparable, and 4 infrastructure failure.

## v0.3.0 to v0.4.0

- Use `llmbench stress --concurrency 1,4,8,16` for warmed concurrency curves and fixed
  arrival-rate profiles. (`llmbench sweep` was merged into `stress`.)
- Code, browser, OS, and agent tasks require the remote executor. They never fall back locally.
- Configure executor keys through environment variables and submit only short-lived task keys.

## v0.4.0 to v0.5.0

- Install benchmark assets as optional `quanttrio-llmbench-data-*` wheels.
- Run `llmbench data verify` after installation and `llmbench coverage` before a suite.
- Configure chat, multimodal, embedding, Judge, and agent targets independently.
- Stable executable coverage is 20 categories. HLE and Fiction.liveBench remain catalog-only.

## v0.5.0 to v0.6.0

- Validate completed runs with `llmbench validate RUN_DIR` before comparison or archival.
- HTML reports are self-contained and add category, sweep, and paired-output visualizations.
- Build outputs include an SBOM; verify release files against `SHA256SUMS`.

## v0.6.0 to v1.0.0

- No report or configuration conversion is required; schema version 2 is the stable 1.x contract.
- Core and executor examples use the `1.0.0` image tag. Keep using integrity-verified v0.5.0 data
  wheels unless a benchmark-specific update is required.
- Stable coverage is 20 categories. The two catalog-only entries are intentionally not counted as
  unsupported or zero-scored categories.
- Review [STABLE_API.md](STABLE_API.md) before depending on implementation-only Python modules.

## Re-running old baselines

For a CI-grade comparison, re-run the old model with the current package, the same installed data
wheel revisions, and the same YAML configuration as the candidate. The comparison command rejects
different data hashes, question keys, prompts, seeds, sampling parameters, or generation settings.
