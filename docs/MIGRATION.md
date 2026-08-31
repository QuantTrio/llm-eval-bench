# Migration guide

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

- Use `llmbench sweep` for warmed concurrency curves and fixed arrival-rate profiles.
- Code, browser, OS, and agent tasks require the remote executor. They never fall back locally.
- Configure executor keys through environment variables and submit only short-lived task keys.

## v0.4.0 to v0.5.0

- Install benchmark assets as optional `quanttrio-llmbench-data-*` wheels.
- Run `llmbench data verify` after installation and `llmbench coverage` before a suite.
- Configure chat, multimodal, embedding, Judge, and agent targets independently.
- Stable executable coverage is 21 categories. HLE and Fiction.liveBench remain catalog-only.

## v0.5.0 to v0.6.0

- Validate completed runs with `llmbench validate RUN_DIR` before comparison or archival.
- HTML reports are self-contained and add category, sweep, and paired-output visualizations.
- Build outputs include an SBOM; verify release files against `SHA256SUMS`.

## v0.6.0 to v1.0.0

- No report or configuration conversion is required; schema version 2 is the stable 1.x contract.
- Core and executor examples use the `1.0.0` image tag. Keep using integrity-verified v0.5.0 data
  wheels unless a benchmark-specific update is required.
- Stable coverage is 21 categories. The two catalog-only entries are intentionally not counted as
  unsupported or zero-scored categories.
- Review [STABLE_API.md](STABLE_API.md) before depending on implementation-only Python modules.

## Re-running old baselines

For a CI-grade comparison, re-run the old model with the current package, the same installed data
wheel revisions, and the same YAML configuration as the candidate. The comparison command rejects
different data hashes, question keys, prompts, seeds, sampling parameters, or generation settings.
