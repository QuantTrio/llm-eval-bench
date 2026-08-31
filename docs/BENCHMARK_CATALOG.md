# Benchmark selection and support boundary

`src/llmbench/data/benchmark_catalog.json` contains a curated 155-benchmark, 22-category snapshot
based on DataLearner on 2026-08-28. Each entry includes its public report count, category, metric,
problem count, institution, and source links. Refresh it with:

```bash
python scripts/snapshot_datalearner_catalog.py
```

## Selection rule

A dataset is bundled only when all of the following hold:

1. It is relevant to text-model quantization regression through one OpenAI-compatible chat API.
2. Its public data and redistribution license are clear.
3. It can be scored locally and objectively without an additional judge model.
4. A useful deterministic subset can fit in the wheel.
5. Within a category, higher DataLearner report count is preferred when the above conditions tie.

The bundled set covers Comprehensive, Science & Reasoning, Math Reasoning, Common Sense,
Truthfulness, and Reading Comprehension. Every report keeps those categories separate.

## Catalog-only categories

- Programming and software engineering: official scores require execution sandboxes, repository
  fixtures, or live task updates. Running model-generated code on the user's machine is not a safe
  default.
- Agent and tool-use: Terminal-Bench, MCP, OSWorld, search, and similar tasks require their own
  environment and harness, not just `/chat/completions`.
- Multimodal: image, video, PDF, and GUI assets require a multimodal request/data pipeline.
- Writing and preference: Elo or judge-model results are not equivalent to deterministic local
  scoring.
- Private, gated, or live benchmarks: data cannot be redistributed in an offline wheel.
- Long context: AA-LCR and LongBench v2 require large document bundles and specialized judging;
  they remain catalogued until a faithful scorer is included.
- Instruction following: IFBench has a task-specific verifier and fewer than 100 published items;
  it remains catalogued until its complete harness can be shipped faithfully.

`llmbench list-benchmarks --top 20` shows the most reported entries; add `--bundled-only` to see
the implemented subset.
