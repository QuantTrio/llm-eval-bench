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

License texts and sources:

- Apache License 2.0: <https://www.apache.org/licenses/LICENSE-2.0>
- MIT License: <https://opensource.org/license/mit>
- Creative Commons Attribution 4.0: <https://creativecommons.org/licenses/by/4.0/>
- Creative Commons Attribution-NonCommercial-ShareAlike 4.0:
  <https://creativecommons.org/licenses/by-nc-sa/4.0/>
- C-Eval data license: <https://github.com/hkust-nlp/ceval/blob/main/LICENSE-DATA>
- GPQA dataset license: <https://github.com/idavidrein/gpqa/blob/main/dataset.zip>

Attribution and citation information remains available from each upstream project. Users are
responsible for ensuring their intended use satisfies all applicable dataset terms, particularly
the C-Eval non-commercial restriction.

