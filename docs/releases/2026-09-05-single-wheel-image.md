# Single-wheel image evaluation acceptance — 2026-09-05

## Delivery contract

Distribution: `llm-bench`, version `2.0.0.dev1`; executable/import: `llmbench`.
One wheel contains the evaluator and all data. `[image]` selects the optional Pillow
dependency, not a second data package or an image download. Base installation includes
the image bytes but does not import/require Pillow to evaluate text.

| Bundled content | Count |
|---|---:|
| Text evaluation questions (eight existing datasets) | 3,300 |
| Synthetic stress prompts | 10 |
| MMMU visual questions | 500 |
| MMMU multiple-choice / open questions | 476 / 24 |
| Deduplicated PNG files / references | 535 / 541 |

All eight text JSONL files retain their prior hashes. Image conversion decodes the old
inline base64 only, preserves exact PNG bytes, and stores 188,745,878 bytes in deduplicated
files. The 500-record MMMU JSONL is 491,059 bytes, SHA-256
`383759d081cab5fd353cfb43568a3595e32c7068e0e3812473bbf5f0fd41a8e6`.

The MMMU source revision is `98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68`; original selection,
license and source JSONL hash remain in the manifest. C-Eval's non-commercial restriction
remains in the data notices and combined distribution license expression.

## Implementation

- The primary `run` command now handles canonical still-image inputs over all four adapters.
  Chat/Responses/Messages/Gemini receive their respective native image content blocks.
- Complete prompts include multiple-choice options. Empty-choice MMMU records are open
  questions and use a local MMMU-style native scorer, including string/numeric aliases.
- Scoring is local and does not require a judge. The open-answer scorer is not claimed to
  reproduce the full official evaluator exactly; numerical answers are asked to include
  decimals, and symbolic algebraic equivalence is not evaluated.
- Pillow loads only when images are selected. Input validation checks actual format,
  dimensions, file integrity, SHA-256, package path boundaries and configured size limits.
- Prompt fingerprints include actual image contents. Resume rejects changed images; input
  hashes are checked again before generation. Base64 for the whole dataset is not cached.
- Reports retain image hashes, dimensions and MIME type alongside question-level results.
- Core dataset IDs take precedence over stale optional packs. No legacy data files were removed.

## Real API verification

Model explicitly selected: `deepseek-v4-flash-vision-exp`; target: `api.deepseek.com`.
Thinking was disabled for this bounded visual acceptance. Native image formats follow
the [DeepSeek vision guide](https://api-docs.deepseek.com/zh-cn/guides/vision/).

The initial protocol probe used two changing image-only digit controls plus one single-image
and one multi-image MMMU question. All 18 requests completed without protocol errors; all
six digit controls were correct. The real MMMU questions scored 8/12 across the repeated
protocol variants. These are compatibility samples, not an official accuracy estimate.

The integrated CLI then tested a single-image choice question, a multi-image choice question
and an open question under Chat/Responses/Messages, both JSON and SSE. All 18 calls completed,
with native per-question scores and image metadata. Review found a false negative where an
explicit decimal in a final answer was discarded by the open-answer parser; it was fixed
before final delivery and captured in regression tests. Earlier run folders remain preserved.

## Packaging acceptance

Checks cover a bare wheel installation without Pillow and an `[image]` installation in two
clean virtual environments. Imports are verified from each environment's `site-packages`.
The bare installation runs a text evaluation and explains the missing extra before any
image request. The image installation validates all 500 questions / 541 image references.

The wheel verifier independently checks all record counts, original text hashes, every PNG
hash, corpus provenance, optional dependency markers, and absence of a separate data package.
The installed wheel is also exercised against the real vision endpoint.

All artifacts and logs are under `runs/vision-single-wheel-20260905.XP0iKu/`.

Final acceptance:

| Check | Result |
|---|---|
| Full automated regression | 460 passed in 13.14 seconds |
| Ruff lint / formatting / diff checks | Passed |
| Core text | 3,300 questions; all eight previous file hashes unchanged |
| Image data | 500 questions / 535 PNGs / 541 references; every hash verified |
| Bare wheel installation | Text evaluation passes with Pillow absent |
| Same wheel plus `[image]` | All image inputs decode and validate; `pip check` passed |
| Wheel/source parity | All 26 Python modules exactly match source |
| Final installed-wheel live Responses SSE | 3/3 calls successful; 3/3 sample answers scored correctly |

The post-fix source smoke scored 2/3, while the final installed-wheel run scored 3/3. These
tiny samples vary and must not be interpreted as performance improvements from packaging.
The explicit-decimal answer is now captured in both cases. All earlier candidate artifacts
remain preserved; only the final path below is the handoff artifact.

Final wheel: `runs/vision-single-wheel-20260905.XP0iKu/dist-final/llm_bench-2.0.0.dev1-py3-none-any.whl`

Size: **188,469,296 bytes** (about 180 MiB).

SHA-256: `0adbe626b8d2dbe1d035279ce6a7667dd696011a7be788b309f15a4c0ffd789a`

After downloading that file:

```bash
python -m pip install './llm_bench-2.0.0.dev1-py3-none-any.whl'
python -m pip install './llm_bench-2.0.0.dev1-py3-none-any.whl[image]'
```

`llm-bench[image]` is the intended package/extra name. No index release has been published,
so these verified commands install the local file, not an unrelated PyPI distribution.

Evidence relative to the artifact directory:

- `visual-probe.json`: 18 native protocol calls, including six image-only controls.
- `cli-visual/results.json`: initial six-way integrated CLI matrix.
- `cli-visual-final/results.json`: post-fix source smoke.
- `wheel-live-final/responses-sse/report.html`: final installed-wheel live report.
- `source-tests-final.xml`: final full-suite JUnit result.
- `check_install.py`: isolated bare/extra installation checks.
- `dist-final/SHA256SUMS`: checksum of the final wheel.

## Limits

Tested locally on Python 3.12 / macOS ARM64. Official OpenAI, Anthropic, xAI and Google
accounts were not used; DeepSeek's compatible routes provide the live evidence here.
Gemini's image encoding has automated local coverage but no Google-hosted live run.
The 500 MMMU and 3,300 text records were packaged and verified; they were not all evaluated
against a live model. Animation, video and image embeddings are outside this image extra.
No API key was written to the repository or test artifacts; no GitHub/PyPI upload was made.
