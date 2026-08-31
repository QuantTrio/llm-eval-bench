# llm-eval-bench

`llmbench` measures answer quality and serving performance through any OpenAI-compatible
`/v1` API. It is designed for repeatable FP16/BF16 versus quantized-model regression runs:
the benchmark samples ship inside the wheel, requests run concurrently, and every run emits
raw responses plus JSON, Markdown, and HTML reports.

## Install

Install the tagged release in one command:

```bash
python -m pip install "quanttrio-llmbench @ https://github.com/QuantTrio/llm-eval-bench/releases/download/v0.2.0/quanttrio_llmbench-0.2.0-py3-none-any.whl"
```

Or install the current `main` branch:

```bash
python -m pip install git+https://github.com/QuantTrio/llm-eval-bench.git
```

Python 3.10 or newer is required. No dataset is downloaded at install or runtime.

## Quick start

```bash
export OPENAI_BASE_URL=https://api.moonshot.cn/v1
export OPENAI_API_KEY=sk-xxx

llmbench list-models
llmbench list-datasets

llmbench run \
  --limit 100 \
  --concurrency 32 \
  --temperature 0 \
  --output-dir runs/candidate-model
```

For a local server that does not require authentication, use `OPENAI_API_KEY=EMPTY`.

`GET /models` selects the only available model automatically. If a server exposes several,
the first is used with a warning; pass `--model MODEL_ID` to make the choice explicit.

## Three run modes

| Command | Purpose | Default concurrency | Scoring |
|---|---|---:|---|
| `llmbench eval` | Stable quality baseline | 1 | Yes |
| `llmbench stress` | Throughput and latency load test | 64 | No |
| `llmbench run` | Quality under concurrent load | 16 | Yes |

Streaming is enabled by default for `run` and `stress`, so the report includes TTFT and TPOT.
Use `--no-stream` for servers without streaming support. Retries, timeouts, HTTP errors,
truncation, empty output, and parse failures remain visible in raw results and summaries.

### Command and parameter reference

```bash
# Deterministic quality evaluation: 100 records from each selected dataset
llmbench eval --dataset mmlu-pro,gpqa-diamond,ceval --limit 100 --concurrency 1

# Random but reproducible 200-record sample per dataset, with repeated generations
llmbench eval --dataset gsm8k,drop --sample 200 --seed 42 --n-samples 3

# Quality and serving performance together
llmbench run --limit 100 --concurrency 32 --stream

# Fixed-duration or fixed-request load tests
llmbench stress --duration 60 --concurrency 64 --max-tokens 128
llmbench stress --requests 500 --concurrency 64 --max-tokens 128
```

| Parameter | Meaning |
|---|---|
| `--base-url` | OpenAI-compatible API root; defaults to `OPENAI_BASE_URL`. |
| `--api-key` | API key; defaults to `OPENAI_API_KEY`; `EMPTY` is allowed. |
| `--model` | Explicit model ID; omitted means discovery through `GET /models`. |
| `--dataset` | Comma-separated bundled IDs and/or local JSONL paths. |
| `--limit N` | First `N` records from each dataset. |
| `--sample N --seed S` | Deterministically sample `N` records from each dataset. |
| `--concurrency N` | Maximum in-flight requests. |
| `--n-samples N` | Generations per question; enables pass@k and consistency metrics. |
| `--temperature`, `--top-p` | Sampling settings sent to the server. |
| `--max-tokens` | Maximum generated tokens; defaults to 4096 or a higher dataset recommendation. |
| `--stream` / `--no-stream` | Enable or disable SSE; streaming enables TTFT/TPOT. |
| `--timeout`, `--retries`, `--retry-backoff` | Request failure policy. |
| `--duration` / `--requests` | Stress-test stopping condition. |
| `--output-dir` | Run artifact directory; generated automatically when omitted. |
| `--memory-gb` | Optional measured model-memory value recorded in the report. |
| `--checkpoint-every` | Durably sync after this many completed requests; defaults to 1. |
| `--progress-interval` | Seconds between structured progress updates; defaults to 5. |
| `--resume DIR` | Strictly resume an interrupted run from its saved manifest. |
| `--request-extra-body JSON` | Add service-specific request fields; core fields cannot be replaced. |

Run `llmbench COMMAND --help` for the authoritative options and defaults.
When `--sample` is present it takes precedence over the default or explicit `--limit`.

### Progress and resume

Every completed request is appended immediately, so interruption does not discard prior work.
Resume uses the saved dataset, model, prompt, and generation fingerprint:

```bash
llmbench run --resume runs/candidate-model
```

Changing the dataset, model, seed, sampling configuration, or request body requires a new run.

## Offline benchmark bundle

The default run uses all eight scored datasets with the first 100 samples from each. `--limit`
is applied per dataset. The bundle sizes are intentionally 100, 200, or 500 records, matching
fast regression use rather than claiming a full official benchmark score.

| Dataset | Category | Bundled | Metric | DataLearner reports* |
|---|---|---:|---|---:|
| GPQA Diamond | Science & Reasoning | 100 | accuracy | 225 |
| MMLU-Pro | Comprehensive | 500 | accuracy | 133 |
| GSM8K | Math Reasoning | 500 | exact match | 70 |
| C-Eval | Comprehensive, Chinese | 500 | accuracy | 48 |
| DROP | Reading Comprehension | 500 | token F1 | 9 |
| TruthfulQA MC1 | Truthfulness | 200 | accuracy | 4 |
| HellaSwag | Common Sense Reasoning | 500 | accuracy | 3 |
| MMLU-Redux | Comprehensive | 500 | corrected-key accuracy | n/a |

\* Report counts are a 2026-08-28 snapshot used only to prioritize broadly reported datasets.
They do not imply that this tool reproduces every result listed by DataLearner.

`llmbench list-benchmarks` exposes the complete 157-entry, 23-category DataLearner snapshot.
Use `--category`, `--top`, or `--bundled-only` to inspect the selection. Benchmarks requiring
a code sandbox, browser/tools, a multimodal pipeline, private data, human preference votes, or
a separate judge model remain catalogued but are not presented as supported scores.

C-Eval data is CC BY-NC-SA 4.0 and is restricted to non-commercial use. See
[THIRD_PARTY_DATASETS.md](THIRD_PARTY_DATASETS.md) before distributing or using bundled data.

## Reports

Every run writes:

```text
summary.json
raw_results.jsonl
events.jsonl
run_manifest.json
run_state.json
report.md
report.html
```

Quality is reported separately by dataset, benchmark category, and question type. Performance
includes QPS, successful QPS, input/output tokens per second, latency p50/p90/p95/p99,
TTFT/TPOT distributions, timeout/error rates, retry attempts, and truncated responses. API keys
are never written to reports.

### Expected result

A successful command prints the selected model and the four artifact paths, for example:

```text
Model: candidate-model
Requests: 800  Successful: 800  Score: 0.7425
Summary: runs/candidate-model/summary.json
Raw results: runs/candidate-model/raw_results.jsonl
Markdown: runs/candidate-model/report.md
HTML: runs/candidate-model/report.html
```

The numbers above are illustrative, not a claimed model score. The actual `summary.json` includes
independent results for each dataset category and question type:

```json
{
  "quality": {
    "sample_mean_score": 0.7425,
    "composite_score": null,
    "quality_valid": true,
    "by_category": {
      "Comprehensive": {"macro_mean_score": 0.78, "samples": 300},
      "Math Reasoning": {"macro_mean_score": 0.71, "samples": 100},
      "Reading Comprehension": {"macro_mean_score": 0.67, "samples": 100}
    },
    "by_question_type": {
      "multiple_choice": {"score": 0.78, "samples": 600},
      "math": {"score": 0.71, "samples": 100},
      "f1": {"score": 0.67, "samples": 100}
    }
  },
  "performance": {
    "qps": 12.4,
    "latency_ms": {"p50": 1820.0, "p95": 3910.0},
    "ttft_ms": {"p50": 210.0, "p95": 480.0},
    "error_rate": 0.0,
    "truncation_rate": 0.0
  }
}
```

A meaningful FP16/BF16-versus-quantized comparison uses identical datasets, prompts, sampling
parameters, concurrency, and server configuration; `llmbench compare` then reports the
regression explicitly.

Compare a baseline and quantized candidate:

```bash
llmbench compare \
  --baseline runs/baseline-fp16/summary.json \
  --candidate runs/candidate-quantized/summary.json \
  --report reports/quantized-vs-fp16.html
```

The comparison reports per-dataset absolute/relative score changes, throughput and QPS changes,
p95 latency reduction, error-rate change, and memory reduction when both runs specify
`--memory-gb`.

## Custom local datasets

Pass any local JSONL path in `--dataset`; comma-separated built-ins and paths may be mixed.

```json
{"id":"q1","dataset":"custom","type":"multiple_choice","question":"...","choices":{"A":"...","B":"..."},"answer":"B"}
{"id":"q2","dataset":"custom","type":"math","question":"...","answer":"#### 42"}
{"id":"q3","dataset":"custom","type":"exact_match","question":"...","answer":"Paris"}
{"id":"q4","dataset":"custom","type":"f1","question":"Passage... Answer:","answer":"red fox"}
```

Use `--sample N --seed 42` for deterministic random selection, or `--limit N` for the first N
records. Raw prompts and outputs are retained for regression diagnosis.

## Development

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/pytest
.venv/bin/python -m build
.venv/bin/python scripts/verify_package.py dist/*.whl
```

Dataset regeneration is maintainer-only and requires `.[data]`. Runtime users never need
`pyarrow` or network access.

The code is Apache-2.0. Bundled benchmark data retains its upstream license.
