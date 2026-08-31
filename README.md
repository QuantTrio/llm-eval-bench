# llm-eval-bench

`llmbench` measures answer quality and serving performance through any OpenAI-compatible
`/v1` API. It is designed for repeatable FP16/BF16 versus quantized-model regression runs:
the benchmark samples ship inside the wheel, requests run concurrently, and every run emits
raw responses plus JSON, Markdown, and HTML reports.

## Install

Recommended: install the complete framework and all publicly redistributable data assets from one
wheel:

```bash
python -m pip install "quanttrio-llmbench-full @ https://github.com/QuantTrio/llm-eval-bench/releases/download/v1.0.1/quanttrio_llmbench_full-1.0.1-py3-none-any.whl"
```

Or install the current `main` branch:

```bash
python -m pip install git+https://github.com/QuantTrio/llm-eval-bench.git
```

The `py3-none-any` Release uses one wheel across all supported Python 3 versions, operating
systems, and CPU architectures. v1.0.1 currently declares a Python 3.10 runtime floor; this is a
source-language requirement rather than a wheel ABI or platform restriction. No dataset is
downloaded at install or runtime.

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

# Warmed concurrency sweep with separate reports per level
llmbench sweep --concurrency 1,4,8,16,32,64 --warmup-requests 20 --requests 200
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
| `--request-rate`, `--ramp-seconds` | Optional fixed arrival rate and gradual worker start. |
| `--prompt-profile` | `short`, `medium`, `long`, or bundled `mixed` stress prompts. |
| `--output-dir` | Run artifact directory; generated automatically when omitted. |
| `--memory-gb` | Optional measured model-memory value recorded in the report. |
| `--checkpoint-every` | Durably sync after this many completed requests; defaults to 1. |
| `--progress-interval` | Seconds between structured progress updates; defaults to 5. |
| `--resume DIR` | Strictly resume an interrupted run from its saved manifest. |
| `--request-extra-body JSON` | Add service-specific request fields; core fields cannot be replaced. |

Run `llmbench COMMAND --help` for the authoritative options and defaults.
When `--sample` is present it takes precedence over the default or explicit `--limit`.

Run the same command from YAML; explicit CLI options and `OPENAI_*` environment variables take
precedence over YAML values:

```bash
llmbench run --config examples/bench.yaml --concurrency 32
```

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

## Full wheel, optional data wheels, and capability suites

The stable executable matrix covers 21 benchmark categories. The recommended v1.0.1 full wheel
combines the complete framework, core data, and all 13 data assets or task descriptors that can be
redistributed publicly. It is one real wheel, not a dependency-only metapackage, and provides
`16/21` category coverage by itself. The remaining five packs must be built locally because of
upstream licensing or mixed assets.

### Included 16-dataset representative matrix

The full wheel has 16 representative datasets available immediately after installation, covering
16 of the 21 stable capability categories. Execution still requires the target shown in the table.
Record counts are the packaged regression set or official complete set; they are not padded when
an official set contains fewer than 100 records.

| Category | Representative dataset | ID | Records | Required target | Native metric |
|---|---|---|---:|---|---|
| AI Agent - 信息收集 | BrowseComp | `browsecomp` | 100 | agent/browser | Judge accuracy |
| AI Agent - 工具使用 | Terminal-Bench 2.0 | `terminal-bench-2` | 89 | agent/executor | success rate |
| Agent 能力评测 | Aider Polyglot | `aider-polyglot` | 200 | agent/executor | pass rate@1 |
| OpenClaw 综合能力 | PinchBench | `pinchbench` | 23 | agent/executor | success rate |
| 代码能力 | HumanEval | `humaneval` | 100 | agent/executor | pass@1 |
| 多模态理解 | MMMU | `mmmu` | 500 | multimodal | accuracy |
| 常识推理 | SimpleBench | `simple-bench` | 10 | chat | accuracy |
| 常识问答 | SimpleQA | `simpleqa` | 500 | chat + Judge | Judge accuracy |
| 指令跟随 | IFBench | `ifbench` | 200 | chat/verifier | prompt loose accuracy |
| 数学推理 | AIME 2025 | `aime-2025` | 30 | chat | exact match |
| 文本向量检索 | MTEB retrieval mini | `mteb-retrieval-mini` | 500 | embedding | recall@1 |
| 真实性评估 | TruthfulQA | `truthfulqa` | 200 | chat | accuracy |
| 科学与综合推理 | GPQA Diamond | `gpqa-diamond` | 100 | chat | accuracy |
| 综合能力 | AGIEval | `agieval` | 200 | chat | accuracy |
| 长上下文能力 | LongBench v2 | `longbench-v2` | 200 | chat + Judge | accuracy |
| 阅读理解 | DROP | `drop` | 500 | chat | token F1 |

These 16 representatives contain 3,452 packaged records. The full wheel also includes MMLU-Pro,
GSM8K, C-Eval, HellaSwag, and MMLU-Redux as supplementary general regression datasets; those five
do not add new categories to the `16/21` capability coverage figure.

### Install the single full wheel

The full wheel is about 235 MB compressed and contains about 492 MB of framework and data files;
allow at least 1 GB for the virtual environment and extracted assets. This method requires the
[GitHub CLI](https://cli.github.com/). Install it into a clean environment so another llmbench
distribution does not own the same import files.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

mkdir -p llmbench-wheel

gh release download v1.0.1 \
  --repo QuantTrio/llm-eval-bench \
  --pattern 'quanttrio_llmbench_full-1.0.1-py3-none-any.whl' \
  --dir llmbench-wheel \
  --clobber

python -m pip install llmbench-wheel/quanttrio_llmbench_full-1.0.1-py3-none-any.whl

python -c 'import llmbench; print(llmbench.__version__)'
python -m pip list --format=freeze | grep '^quanttrio-'
llmbench data list
llmbench data verify
llmbench coverage
```

Expected final lines include version `1.0.1`, every installed pack with `status=ok`, and:

```text
quanttrio-llmbench-full==1.0.1
Coverage: 16/21 categories
```

The `py3-none-any` compatibility tag means the same pure-Python wheel is used across all supported
Python 3 minor versions and platforms. The current `Requires-Python: >=3.10` metadata records the
runtime syntax baseline; it does not require a CPython-version-specific or platform-specific wheel.

The five local-build-only packs are Creative Writing v3, MMEB-v2 Image, GDPval Gold,
LiveCodeBench, and SuperGLUE. Their pinned builders are under `data-packs/` in the source archive.
After reviewing their upstream terms and obtaining any required source access, build and install
those local-only packs to reach `21/21`. Each restricted pack contains its own README with source,
license, and preparation requirements.

The supported public installation is only the v1.0.1 full wheel. The projects under `data-packs/`
are maintainer build inputs and are not separate public Release artifacts.

### Configure and start evaluation

Create `bench.yaml`. This minimal configuration runs chat and multimodal-compatible datasets
through one OpenAI-compatible model and uses the same endpoint as a smoke-test Judge. For a formal
comparison, use a separate Judge endpoint. Remove the `multimodal` block if the model does not
support OpenAI content parts.

```yaml
schema_version: 2

targets:
  chat:
    base_url: http://YOUR_LLM_HOST:PORT/v1
    model: your-model-id
    api_key_env: CHAT_API_KEY
  multimodal:
    base_url: http://YOUR_LLM_HOST:PORT/v1
    model: your-model-id
    api_key_env: MULTIMODAL_API_KEY

judge:
  base_url: http://YOUR_LLM_HOST:PORT/v1
  model: your-model-id
  api_key_env: JUDGE_API_KEY
  repeats: 3

run:
  concurrency: 16
  temperature: 0
  top_p: 1
  max_tokens: 8192
  timeout: 300
  retries: 2
  retry_backoff: 2
  seed: 42
  stream: true
  checkpoint_every: 1
  progress_interval: 5
```

Export the keys. Use `EMPTY` for an endpoint that does not require authentication:

```bash
export CHAT_API_KEY=EMPTY
export MULTIMODAL_API_KEY=EMPTY
export JUDGE_API_KEY=EMPTY
```

First run a five-record-per-dataset smoke suite:

```bash
llmbench suite \
  --config bench.yaml \
  --limit 5 \
  --output-dir runs/suite-smoke \
  2>&1 | tee suite-smoke.log
```

Then start the 100-record-per-dataset evaluation:

```bash
llmbench suite \
  --config bench.yaml \
  --limit 100 \
  --output-dir runs/suite-100 \
  2>&1 | tee suite-100.log
```

`--limit` is applied to each dataset, not to the suite as a whole. Official sets containing fewer
records use their complete set without duplication. Progress is visible in the terminal and in
`events.jsonl`; completed requests are appended immediately to `raw_results.jsonl`.

Chat, multimodal, embedding, Judge, and remote-agent targets are configured independently. A
missing target is reported as `unsupported_capability` and is excluded from scoring rather than
silently counted as a wrong answer. To run the embedding and agent categories, add the `embedding`
and `agent` targets from [examples/bench.yaml](examples/bench.yaml) and deploy the remote executor
described in [docs/EXECUTOR.md](docs/EXECUTOR.md). Runtime remains offline after wheel installation,
except for benchmarks such as BrowseComp whose task definition requires controlled internet access.
See [docs/SUPPORTED_BENCHMARKS.md](docs/SUPPORTED_BENCHMARKS.md).

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

Validate the machine-readable artifacts before archiving or comparing a run:

```bash
llmbench validate runs/candidate-model
```

Quality is reported separately by dataset, benchmark category, and question type. Performance
includes QPS, successful QPS, input/output tokens per second, latency p50/p90/p95/p99,
TTFT/TPOT distributions, timeout/error rates, retry attempts, and truncated responses. API keys
are never written to reports. The self-contained HTML report includes a category radar, dataset
bars, and question-level prompt/output/parse details. Sweep reports include offline QPS and p95
latency curves; comparison reports expose paired baseline and candidate raw outputs.

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
  --baseline runs/baseline-fp16 \
  --candidate runs/candidate-quantized \
  --policy examples/regression.yaml \
  --report reports/quantized-vs-fp16.html
```

Comparison requires matching dataset hashes, request keys, prompts, seeds, sampling settings, and
generation settings. It emits a paired result JSONL, per-dataset changes, correct-to-wrong
transitions, and deterministic 95% bootstrap confidence intervals. Exit codes are `0` for pass,
`2` for a regression-policy failure, `3` for incomparable runs, and `4` for infrastructure or
artifact errors.

## Stable compatibility contract

v1.0 freezes the documented CLI commands, YAML schema v2, report schema v2, comparison exit
codes, and the 21-category capability matrix. Future 1.x releases may add optional fields and
commands but will continue to read valid v1.0 artifacts. The v1.0.1 full wheel embeds the stable
public data-pack contract and assets in the same installation artifact.

See [docs/STABLE_API.md](docs/STABLE_API.md) for the complete stability and deprecation policy.

## Remote executor

Install the service dependencies and start the API:

```bash
python -m pip install 'quanttrio-llmbench[executor]'
llmbench executor serve --config executor/executor.yaml
```

Validate a deployment with an ephemeral task key:

```bash
export EXECUTOR_URL=https://executor.example.com
export EXECUTOR_TASK_KEY=short-lived-secret
llmbench executor smoke --image quanttrio/llmbench-sandbox:1.0.1
```

Executor jobs run in allowlisted, read-only remote containers with CPU, memory, PID, output, and
time limits. Network is disabled by default; enabled jobs join an internal-only network and use
an allowlisted egress proxy. Temporary keys are held only for the task and redacted from stored
requests, events, errors, and artifacts. See [docs/EXECUTOR.md](docs/EXECUTOR.md).

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

See [docs/MIGRATION.md](docs/MIGRATION.md) when moving an existing run or CI pipeline from a
v0.1.x release.

The code is Apache-2.0. Bundled benchmark data retains its upstream license.
