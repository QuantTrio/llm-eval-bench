# DeepSeek thinking and interrupted-process resume — 2026-09-05

Model: `deepseek-v4-flash`. Installed package: `2.0.0.dev0`.
Environment: `runs/deepseek-live-20260905.m2BsK6/venv` (Python 3.12.14, macOS ARM64).
This extends the earlier live DeepSeek compatibility acceptance with real thinking-mode
requests and operating-system process interruption.

## Thinking-mode acceptance

| Protocol | JSON | SSE | Reasoning tokens reported (JSON / SSE) |
|---|---|---|---|
| Chat Completions | 2/2 successful | 2/2 successful | 284 / 311 |
| Responses | 2/2 successful | 2/2 successful | 158 / 204 |
| Anthropic Messages | 2/2 successful | 2/2 successful | unavailable / unavailable |

Each case evaluated the same 1 MMLU-Pro and 1 GSM8K record, concurrency 1, retries 0,
4096 output tokens, 180-second request timeout. All 12 answers scored correctly, with no
unexpected truncation or request errors. These repeated small samples are compatibility
evidence, not a full benchmark or a reliable quality comparison.

Thinking settings follow the [official guide](https://api-docs.deepseek.com/zh-cn/guides/thinking_mode/):

- Chat: `thinking.type=enabled`, `reasoning_effort=low`.
- Responses: `reasoning.effort=low`.
- Messages: `thinking.type=enabled`, `output_config.effort=low`.
- Temperature and top-p explicitly null in YAML, so they are omitted from requests.

A passive observer wrapped the installed client's request and SSE methods. It did not
change request bodies, protocol parsing, scoring, or checkpoint writing. It recorded only
request hashes, IDs, event kinds, text lengths and timing; no headers or raw thought text.

SSE observation found 1499 / 840 / 610 thinking characters for Chat / Responses / Messages.
In each stream, thinking appeared before final answer text. Answer-delta character counts
matched the scored `raw_output` lengths, confirming thought deltas were not added to answers.
The Messages API returned thinking blocks but no separate thinking-token counter; the
report correctly retains that metric as unavailable. TPOT remains unavailable for these
thinking runs instead of using hidden thinking tokens as answer tokens.

## Actual process interruption and resume

| Protocol | Signal | Saved before interruption | Newly requested on resume | Final unique results |
|---|---|---:|---:|---:|
| Chat | SIGINT (Ctrl+C equivalent) | 2 | 2 | 4 |
| Responses | SIGINT | 2 | 2 | 4 |
| Messages | SIGINT | 2 | 2 | 4 |
| Responses | SIGKILL | 2 | 2 | 4 |

All four cases used thinking mode and streaming, with 2 MMLU-Pro and 2 GSM8K records.
Each process was interrupted after confirming two saved records, then a fresh CLI process
resumed using only `--resume RUN_DIR` plus the same credential environment reference.

Checks passed:

- Saved model ID, protocol, dataset hashes, prompt hashes, and generation settings restored.
- Manifest unchanged; both saved rows retained every original field and request ID.
- Exactly two requests started during resume; neither matched an already completed prompt.
- Final count exactly four unique question/sample keys, with no request errors.
- A second resume after completion generated zero additional requests.

Each interrupted process had already started its next request. In-flight unfinished work
was eligible for retry; this is not an exactly-once billing guarantee. Across the full
live acceptance there were 35 client request-start events and 31 saved final results,
including four starts cancelled during interruption. Provider billing was not queried.

## Thinking-only truncation

Each protocol was separately tested with an 8-token output budget in streaming mode.
All three produced thinking without a final answer. The application recorded:

- `finish_reason=length`;
- `quality_valid=false`;
- empty final answer and unavailable answer TTFT / TPOT.

These are expected validity failures, not protocol errors or passing model evaluations.

## Automated regression and artifacts

Added [tests/test_process_resume.py](../../tests/test_process_resume.py): an independent
loopback HTTP server pauses mid-SSE while a real CLI child receives SIGINT or SIGKILL.
Its server-side request log proves completed questions are not resent, and persisted
result rows remain byte-for-byte unchanged. Completing a resumed run and resuming again
produces zero new requests.

Full regression: **268 passed in 12.77 seconds**. Ruff lint/format and diff whitespace checks
passed; 117 Python files formatted. No runtime source fix was needed.

All live artifacts are under:

`runs/deepseek-thinking-resume-20260905.j5P1sn/`

- `results.json`: all 13 acceptance cases and assertions.
- `thinking-API-json|sse/run/report.html`: normal thinking reports.
- `resume-API-sigint/run/report.html`: resumed reports.
- `resume-responses-sigkill/run/report.html`: forced-termination report.
- `CASE/requests.jsonl`: passive request and stream observation.
- `CASE/saved-before-resume.json`: original completed rows for resume checks.
- `CASE/run/events.jsonl` and `CASE/console.log`: progress and process logs.
- `offline-regression.xml`: full-suite JUnit results.

The credential entered via an echo-disabled prompt and was only held in process memory
and the child environment. Exact-key scanning of generated files passed. No credential
file, Git commit, push, or release was created.

## Remaining limits

Verified thinking effort `low` only. The live interruption tests used concurrency 1 and
interrupted after durable records; they do not establish behavior during disk failure,
power loss, a torn JSONL write, or high-concurrency cancellation. High/max reasoning,
other providers/models, and other operating systems remain separate acceptance work.
