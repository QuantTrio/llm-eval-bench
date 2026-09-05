# Third-party dataset notices

The Apache-2.0 license for `llmbench` code does not replace the licenses of bundled benchmark
records. Each generated JSONL file is a format conversion and subset of the named upstream data.

| Bundled file | Upstream revision | License | Notes |
|---|---|---|---|
| `mmlu-pro.jsonl` | TIGER-Lab/MMLU-Pro `b189ec7` | Apache-2.0 | First 500 test records after stable ID sort. |
| `mmlu-redux.jsonl` | aryopg/mmlu-redux `4563cfa` | CC BY 4.0 | 500 corrected-key records; invalid corrected choices are excluded. |
| `gpqa-diamond.jsonl` | idavidrein/gpqa `56686c0` | CC BY 4.0 | 100 records; choices are deterministically shuffled. Includes the upstream canary. |
| `gsm8k.jsonl` | openai/grade-school-math `3101c7d` | MIT | First 500 test records. |
| `ceval.jsonl` | ceval/ceval-exam `617524a` | CC BY-NC-SA 4.0 | 500 validation records, balanced round-robin across 52 subjects. Non-commercial use only. |
| `hellaswag.jsonl` | Rowan/hellaswag `218ec52` | MIT | First 500 validation records. |
| `truthfulqa.jsonl` | sylinrl/TruthfulQA `d71c110` | Apache-2.0 | First 200 MC1 records. |
| `drop.jsonl` | OpenAI simple-evals DROP v0 dev | Apache-2.0 | First 500 preprocessed development records. |
| `stress.jsonl` | QuantTrio | Apache-2.0 | Synthetic prompts written for this project. |
| `mmmu.jsonl` and `images/mmmu/*.png` | MMMU/MMMU `98e6ac0` | Apache-2.0 | Existing 500-record validation subset; 476 choice questions and 24 open questions. PNG bytes preserved and deduplicated into 535 images (541 references). |

License texts and sources:

- Apache License 2.0: <https://www.apache.org/licenses/LICENSE-2.0>
- MIT License: <https://opensource.org/license/mit>
- Creative Commons Attribution 4.0: <https://creativecommons.org/licenses/by/4.0/>
- Creative Commons Attribution-NonCommercial-ShareAlike 4.0:
  <https://creativecommons.org/licenses/by-nc-sa/4.0/>
- C-Eval data license: <https://github.com/hkust-nlp/ceval/blob/main/LICENSE-DATA>
- GPQA dataset license: <https://github.com/idavidrein/gpqa/blob/main/dataset.zip>
- MMMU dataset: <https://huggingface.co/datasets/MMMU/MMMU>
- MMMU evaluation rules: <https://github.com/MMMU-Benchmark/MMMU/blob/main/mmmu/utils/eval_utils.py>

MMMU open answers use a local scorer based on the upstream normalization and candidate-matching
rules. It additionally excludes hidden thinking, preserves deterministic candidate order, and
extracts complete grouped/scientific numbers. Scores on this packaged subset are not claimed to
reproduce the official full-dataset evaluation. Data conversion only decodes existing base64 PNGs;
no image resizing, pixel changes, or extra records are introduced.

Attribution and citation information remains available from each upstream project. Users are
responsible for ensuring their intended use satisfies all applicable dataset terms, particularly
the C-Eval non-commercial restriction.
