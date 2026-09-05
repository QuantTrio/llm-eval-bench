# llm-eval-bench

Evaluate model quality, speed, and reliability through hosted or local APIs.
One wheel includes 3,300 text questions, 500 MMMU image questions, and all image assets.

## Usage

Python 3.10+. [Download the wheel](https://github.com/QuantTrio/llm-eval-bench/releases/download/v2.0.0.dev1/llm_bench-2.0.0.dev1-py3-none-any.whl), then install:

```bash
python -m pip install ./llm_bench-2.0.0.dev1-py3-none-any.whl
# Optional images: python -m pip install './llm_bench-2.0.0.dev1-py3-none-any.whl[image]'
```

Run a local model (no key needed for an unauthenticated server):

```bash
llmbench run --base-url http://127.0.0.1:8000/v1 --api chat --model MODEL --output-dir runs/demo
```

Or use a hosted provider:

```bash
export OPENAI_API_KEY=YOUR_KEY
llmbench run --provider openai --model MODEL
```

Set `ANTHROPIC_API_KEY`, `XAI_API_KEY`, or `GEMINI_API_KEY` for the other providers.
Custom endpoints use `LLMBENCH_API_KEY`. An explicit `--base-url` overrides the provider address.

Open `runs/demo/report.html` for results. `summary.json` contains machine-readable scores;
`raw_results.jsonl` keeps individual answers and `events.jsonl` records progress.

```bash
llmbench datasets
llmbench run --resume runs/demo
llmbench compare --baseline runs/model-a --candidate runs/model-b
```

Use `--dataset ID --limit N` to select questions (`N` per dataset).
The default `quick` preset is a smoke test; `--preset standard` evaluates 100 records per
default dataset. Image evaluation uses `--dataset mmmu` with a vision-capable model ID.
See `llmbench run --help` or the [usage guide](docs/USAGE.md) for more options.

## Features

- Bundled offline data and local scoring; no dataset downloads or gateway service.
- Native text/image adapters with bounded concurrency and streaming.
- Per-dataset scores, throughput, latency, first-answer latency, and visible failures.
- Live logs, strict resume, and portable HTML/JSON reports.
- Dataset, prompt, and image hashes for reproducible inputs and comparable runs.

These are packaged sample evaluations, not official full-benchmark scores.
Code is Apache-2.0; data retains its [upstream terms](THIRD_PARTY_DATASETS.md).
C-Eval is restricted to non-commercial use.

## Models and APIs

| Service | Model | API / option |
|---|---|---|
| OpenAI | GPT / reasoning models | `--provider openai`: Responses |
| Anthropic | Claude models | `--provider anthropic`: Messages |
| xAI | Grok models | `--provider xai`: Responses |
| Google | Gemini models | `--provider gemini`: GenerateContent |
| DeepSeek, vLLM, SGLang, custom services | Explicit served model ID | `--base-url URL --api chat\|responses\|messages\|generate-content` |

Select a protocol and model capability actually exposed by the server.
Support covers text and still-image evaluation, with JSON or streaming output.
See [compatibility and verified scope](docs/API_COMPATIBILITY.md).
