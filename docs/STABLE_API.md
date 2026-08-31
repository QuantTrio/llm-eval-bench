# Stable interfaces for llmbench 1.x

This document defines the compatibility contract for the 1.x release line. The command-line and
artifact interfaces listed here are stable; internal Python modules are not a public SDK.

## Command-line interface

The following command families and their documented options are stable:

- `llmbench eval`, `llmbench run`, `llmbench stress`, and `llmbench sweep`
- `llmbench compare --baseline DIR --candidate DIR --policy POLICY`
- `llmbench suite --config CONFIG`
- `llmbench data list`, `llmbench data verify`, and `llmbench coverage`
- `llmbench validate RUN_DIR`
- `llmbench executor serve` and `llmbench executor smoke`

Options may be added in a 1.x release. Existing options will not change meaning without a
deprecation warning in at least one minor release.

## Configuration and secrets

`schema_version: 2` YAML is the stable configuration format. Explicit CLI values take precedence
over the YAML file, followed by environment variables and documented defaults. API and task keys
are read from environment variables and are excluded from configuration snapshots, logs, reports,
events, and artifacts.

Generation length precedence is: explicit CLI value, dataset override, dataset recommendation,
then command default. Quality runs default to 4096 tokens and stress runs default to 128. An
explicit value below a dataset recommendation remains allowed but produces a persisted warning.

## Run and comparison artifacts

New 1.x runs write schema-v2 `summary.json` and `run_manifest.json`, plus append-only
`raw_results.jsonl` and `events.jsonl`. Optional fields may be added, but required v1.0 fields will
remain readable throughout 1.x. Validate a directory with `llmbench validate RUN_DIR`.

Comparison exit codes are stable:

- `0`: passed or no configured regression
- `2`: regression-policy failure
- `3`: runs are not comparable
- `4`: infrastructure or artifact failure

Paired comparison requires identical dataset hashes, request keys, prompts, seeds, sampling, and
generation settings. A heterogeneous set of native metrics is never silently collapsed into an
unweighted accuracy score.

## Capability and data contract

The stable representative matrix contains 21 categories. Chat, multimodal, embedding, Judge, and
agent targets are independent. A missing or incompatible target is reported as `unsupported` and
excluded from score aggregation while reducing coverage.

Data packages use the `llmbench.data.v1` entry-point contract and include a version, source
revision, SHA-256, license metadata, sample count, and selection description. The v1.0.1 full
wheel embeds the stable public data packs with the framework. HLE and Fiction.liveBench are
retained only in the immutable 157-entry reference catalog and are not part of the stable coverage
denominator.

## Executor boundary

Generated or untrusted code never falls back to local execution. Executor tasks use allowlisted
rootless containers with CPU, memory, disk/output, PID, network-domain, and time limits. Temporary
task keys exist only in task memory or the mode-0600 transient environment file and are destroyed
when the task ends.

## Deprecation policy

Breaking CLI, schema, data-entry-point, or executor-protocol changes require a new major version.
Security fixes may reject previously accepted unsafe input without a deprecation window.
