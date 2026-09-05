# 2.0.0.dev0 local preview — 2026-09-05

## Scope

Deliver a small, independent model evaluation CLI with native text adapters for Chat
Completions, Responses, Messages, and GenerateContent. Keep the README to Usage, Features,
and Models and APIs. The primary commands are `run`, `compare`, and `datasets`.

Every new evaluation requires a model ID in CLI/YAML before sending HTTP requests. The
server receives that ID; `/models` is never consulted as part of an evaluation. Resume
uses the saved model and protocol, and rejects conflicting settings.

Provider presets are address/authentication conveniences; all protocols accept custom
Base URLs with preserved prefixes. No authentication is required by the client for an
unauthenticated local server. Protocols do not silently change after an error.

## Implementation decisions

- Reuse one HTTP connection pool, timeout/retry path, SSE reader, runner, scorer, and
  artifact writer. Wire format differences live in one small adapter module.
- Avoid provider SDKs or a gateway dependency. Existing lightweight runtime dependencies
  remain `httpx`, `typer`, `pyyaml`, and `jsonschema`.
- Store answer text and the last provider payload instead of buffering full event streams.
  Record request IDs, cache/reasoning usage, and whether usage was actually reported.
- Measure TTFT at first answer text. Suppress TPOT when reasoning/token accounting is
  insufficient; cross-model token rates retain their provider tokenizer definitions.
- Retain per-dataset native metrics. Errors, parse failures, and truncation affect validity.
- Reject run-directory overwrite; update final raw results atomically. Hash actual prompt
  messages as well as dataset content and request keys.
- Preserve the pre-existing uncommitted refactor of run specifications, sessions, and data
  builders. Existing advanced commands are hidden compatibility entry points, not part of
  the new primary workflow. No stored datasets or user run artifacts were removed.

## Local acceptance

Base commit: `f636db0` plus local working-tree changes. No Git commit, push, tag, or release
was created. Local interpreter: Python 3.12.14 on macOS arm64.

The initial suite contained 82 passing tests. Acceptance extends it with protocol fixture
tests and four APIs times JSON/SSE over real loopback HTTP. Fixtures use the explicit
`fixture-model` ID and known answers; their scores are test evidence, not model benchmarks.

```bash
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/python -m pytest --tb=short
.venv/bin/python -m build --wheel --outdir dist/local-preview
.venv/bin/python scripts/verify_package.py dist/local-preview/quanttrio_llmbench-2.0.0.dev0-py3-none-any.whl
```

Final local acceptance:

| Check | Result |
|---|---|
| Full source suite | 266 passed in 12.30 seconds |
| Lint / formatting / diff whitespace | Passed; 116 Python files formatted |
| Wheel contents | 9 bundled resources and 155 reference catalog entries verified |
| Clean installation | New venv, imports verified from site-packages, `pip check` passed |
| Installed-wheel protocol / endpoint / HTTP tests | 149 passed in 1.09 seconds |
| Model ID enforcement | Missing ID rejected before HTTP; explicit ID never lists models |
| README | 76 lines, three sections; documented CLI examples exercised locally |

The clean wheel environment installed only the package runtime dependencies plus pytest
and pytest-asyncio for acceptance; provider SDKs, FastAPI, and a GPU runtime were not needed.
Later review added tests for resume credential references, explicit null sampling settings,
and contradictory reasoning counters; all are included in the final passing counts above.

Artifacts relative to the repository root:

- `runs/local-preview-20260905.tQNPl9/accepted-tests.xml`: full-suite JUnit results.
- `runs/local-preview-20260905.tQNPl9/wheel-tests.xml`: installed-wheel JUnit results.
- `runs/local-preview-20260905.tQNPl9/example-responses/report.html`: fixture report.
- `runs/local-preview-20260905.tQNPl9/example-responses/events.jsonl`: persisted progress.
- `runs/local-preview-20260905.tQNPl9/dist/quanttrio_llmbench-2.0.0.dev0-py3-none-any.whl`:
  local wheel, 768,237 bytes (about 750 KiB).

Wheel SHA-256: `2b3de51cf36bc7fca864f83a67586fc083e5a00c83c1dee513db76cf753d99cd`.

## Limits

No authenticated hosted-provider or GPU inference-server run was performed. Actual server
versions, model templates, provider availability, and credential routing need a live smoke
test using an explicit model ID. Compatibility targets the text-evaluation subset described
in [API_COMPATIBILITY.md](../API_COMPATIBILITY.md), not every provider feature.

The available local Python interpreter is 3.12. Python 3.10+ remains the declared baseline;
other interpreter/platform executions are not claimed by this local acceptance.
Load mode reports performance only and does not support resume. Scored runs support resume.
The historical large full-data bundle is not rebuilt or published as part of this preview.
