# GitHub preview publication — 2026-09-05

Version: `2.0.0.dev1` (`llm-bench`); GitHub tag: `v2.0.0.dev1`.

This publishes the locally reviewed text/image evaluator and single wheel described in
[the image acceptance record](2026-09-05-single-wheel-image.md). The README has three
sections, uses normal wheel installation by default, and keeps the image extra in a
comment. No PyPI upload is part of this publication.

The release contains one wheel and its `SHA256SUMS`. The wheel includes all 3,300 text
questions, 10 stress prompts, 500 MMMU questions and 535 PNGs. `[image]` only selects
Pillow as an optional dependency; there is no second data wheel or runtime data download.

Before publication, the staged source tree is exported to an isolated directory and
tested without local caches or optional maintainer payloads. Missing legacy source-pack
JSONL files are explicitly skipped; the bundled text/image checks always run.
The wheel is rebuilt from that clean tree and every dataset/image hash is verified.

Public download:

https://github.com/QuantTrio/llm-eval-bench/releases/download/v2.0.0.dev1/llm_bench-2.0.0.dev1-py3-none-any.whl

The existing v1.0.1 release remains available; this new version is a prerelease.
Container publishing is manual-only and is not triggered by this source/tag update.

Local checks before upload:

- Working tree: 460 tests passed.
- Clean staged-tree export: 453 passed, 7 explicitly skipped optional maintainer payloads.
- Lint, formatting, clean sdist/wheel build, data/image verification and SBOM generation passed.
- Public wheel size: 188,468,579 bytes.
- Public wheel SHA-256: `768c8c49a55e17ec84525693f7fdbf4d8e0ab30c017c50084a3952b4b46dc2e7`.

The README change updates wheel metadata, so this published artifact has a different hash
from the earlier locally tested wheel. Runtime modules and all bundled data are unchanged.

The first hosted CI run exposed a test-package import difference between `python -m pytest`
and the `pytest` executable. Declaring `tests` as a package fixes both entry points.
The exact CI command was reproduced in the clean export: 453 passed, 7 optional-data skips.
This fix changes test collection only, not the published wheel's runtime or data.
