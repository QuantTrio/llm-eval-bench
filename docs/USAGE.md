# Usage

Install the local checkout with `python -m pip install .` (Python 3.10+), or `'.[image]'`
to enable image validation. The distribution is `llm-bench`; the command stays `llmbench`.
To build a
distributable wheel, use `python -m build --wheel --outdir dist/local-preview` after
installing the development dependencies. The single wheel includes all text and image data; no
provider SDK, Docker daemon, judge, browser, or dataset download is required for `run`.

## Model and endpoint

Every new evaluation requires an explicit model ID through `--model` or a YAML target.
The ID must match the model served by the endpoint; specifying it bypasses model discovery.
Use the optional diagnostic `llmbench list-models --base-url URL` to inspect advertised IDs.
Servers without a model-list API work with an explicit ID.

## One wheel, optional image dependency

```bash
python -m pip install './llm_bench-2.0.0.dev1-py3-none-any.whl'
python -m pip install './llm_bench-2.0.0.dev1-py3-none-any.whl[image]'
```

Both install exactly the same wheel: 3,300 text questions across eight datasets, 10 stress
prompts, 500 MMMU questions and 535 deduplicated original PNGs. The image extra installs
Pillow only; pip extras cannot conditionally install files inside one wheel. No second
data package or runtime image download is used. The bare installation runs text without Pillow.

These are local file installation commands, not a claim that this name/version is published
on PyPI. Install into a clean virtual environment when moving from older `quanttrio-llmbench`
or `quanttrio-llmbench-full` distributions, which own the same Python import namespace.

```bash
llmbench run --base-url https://api.deepseek.com --api responses \
  --model deepseek-v4-flash-vision-exp --dataset mmmu --limit 5 --output-dir runs/vision
```

Set `LLMBENCH_API_KEY` for authenticated custom endpoints. The standard `run` command handles
both text and images. Use a vision-capable model ID; a text-only model cannot prove image support.
MMMU includes 476 choice questions and 24 open questions; choices and image order are preserved.
Open questions use a documented local MMMU-style scorer, not a claim of official full-score parity.
Reports retain image hashes, MIME types and dimensions. Inputs are validated before requests;
resume rejects changed image contents even if the question IDs and JSONL are unchanged.

Custom JSONL images use `metadata.assets: ["relative-image.png"]`, relative to the JSONL file.
Supported inputs are PNG/JPEG/WebP/static GIF: at most 16 images, 20 MiB per image, 64 MiB total,
and 25 million pixels per image. Animated files and remote image URLs are rejected for this
offline evaluation path. Pixels are not resized or rewritten during inference.

```bash
# Local OpenAI-compatible Chat Completions, without authentication.
llmbench run --base-url http://127.0.0.1:8000/v1 --api chat --model local-model --preset quick

# Select Responses or Messages when your local frontend exposes that endpoint.
llmbench run --base-url http://127.0.0.1:8000/v1 --api responses --model local-model --preset quick
llmbench run --base-url http://127.0.0.1:30000/v1 --api messages --model local-model --preset quick

# Provider shortcuts read their provider-specific API key from the environment.
llmbench run --provider openai --model MODEL_ID --preset quick
llmbench run --provider anthropic --model MODEL_ID --preset quick
llmbench run --provider xai --model MODEL_ID --preset quick
llmbench run --provider gemini --model MODEL_ID --preset quick
```

Keys: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `XAI_API_KEY`, `GEMINI_API_KEY` respectively.
`--api-key` supplies a custom key; unauthenticated custom endpoints need no dummy value.
Provider selection fills in defaults. An explicit `--base-url` always replaces the address:

```bash
llmbench run --provider openai --base-url https://your-proxy.example/team/v1 --model MODEL_ID
```

Use the API root, not the final generation route. A prefix such as `/team/v1` or
`/team/v1beta` is preserved. Only a bare origin gets a default version path appended.
Credentials belong in headers, not URL query strings. `--api` is explicit; no failed
request switches protocol or redirects to another provider.

## Workload

| Option | Meaning |
|---|---|
| `--preset quick` | Small smoke evaluation; not a statistically strong capability estimate |
| `--preset standard` | 100 records from each of six default datasets |
| `--preset full` | All packaged records in the default datasets, not upstream full datasets |
| `--dataset IDs_OR_PATHS` | Comma-separated dataset IDs and/or local JSONL paths |
| `--limit N` | First N records **per dataset** |
| `--sample N --seed S` | Reproducible sample selection per dataset |
| `--mode quality` | Scored evaluation, default concurrency 1 |
| `--mode both` | Scored evaluation with performance metrics, default concurrency 16 |
| `--mode load --requests N` | Fixed-request performance run without answer scoring |
| `--concurrency N` | Bound the number of in-flight generation requests |
| `--max-tokens N` | Output budget; truncation is reported |
| `--no-stream` | Non-streaming response; TTFT/TPOT unavailable |
| `--timeout N --retries N` | Request timeout and transient failure retries |
| `--output-dir DIR` | Explicit destination for all run artifacts |

Temperature and top-p support varies by model. Non-chat protocols omit unspecified
sampling settings; specify supported values when needed. `--seed` always controls local
dataset selection, and is only sent to protocols with a seed field. Hosted generation
is not guaranteed deterministic. `--request-extra-body` supports provider-specific generation
settings; evaluated prompts, model IDs, and unsupported stateful/tool workflows are protected.

```json
{"id":"q1","dataset":"custom","type":"exact_match","question":"What is 6 times 7?","answer":"42"}
```

Store records as JSONL, then pass their path using `--dataset`. Other local scorers support
multiple choice, math, and token F1. Each score retains its metric and dataset identity.

## Results and resume

`report.html` opens offline. `report.md` and `summary.json` contain summaries;
`raw_results.jsonl` contains each request result; `events.jsonl` provides progress;
`run_manifest.json` and `run_state.json` support resume.

```bash
llmbench run --resume runs/demo
llmbench validate runs/demo
llmbench compare --baseline runs/model-a --candidate runs/model-b --report reports/compare.html
```

Resume restores the saved model ID, endpoint, API, and evaluation settings. Explicitly
changing them is rejected. New runs cannot overwrite existing run artifacts.
Comparisons require matching evaluation inputs and settings, including protocol. Configure
optional thresholds with `--policy examples/regression.yaml`. Exit codes: 0 completed/pass,
2 invalid CLI input or policy failure, 3 incomparable runs, 4 infrastructure/artifact failure.
Completed request failures still produce reports. Inspect `quality_valid`, error rates,
parse failures, and truncation alongside scores; load-only runs have no valid quality score.

Advanced 1.x commands remain callable for existing integrations but are hidden from the
primary help. Optional packs and the legacy executor are documented separately in
[SUPPORTED_BENCHMARKS.md](SUPPORTED_BENCHMARKS.md) and [EXECUTOR.md](EXECUTOR.md).

## Local development

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/python -m pytest
.venv/bin/python -m build --wheel --outdir dist/local-preview
```

Protocol tests use local fixtures and loopback HTTP servers, without cloud keys or GPUs.
They verify our adapter behavior; a hosted provider or an actual inference deployment
needs a separate live smoke test using its explicit model ID.
