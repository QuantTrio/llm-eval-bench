# Consolidated public data wheel

`quanttrio-llmbench-data-all` contains the 13 v0.5.0 benchmark data packs whose assets or task
descriptors can be redistributed publicly. It is a real aggregate data wheel, not a metapackage:
all JSONL assets, integrity metadata, source revisions, and license notices are stored inside the
wheel.

The aggregate wheel provides AGIEval, Aider Polyglot, AIME 2025, BrowseComp, HumanEval, IFBench,
LongBench v2, MMMU, MTEB retrieval mini, PinchBench, SimpleBench, SimpleQA, and Terminal-Bench 2.0.
Together with the three representative datasets in the core wheel, coverage is `16/21`.

Do not install this wheel together with the 13 individual public data wheels because they expose
the same dataset IDs. Use the aggregate wheel for the two-wheel installation path, or individual
wheels when only a subset is needed.

Maintainers can rebuild it after generating each source pack:

```bash
python data-packs/all-public/build_pack.py
python -m build data-packs/all-public
```

Creative Writing v3, MMEB-v2 Image, GDPval Gold, LiveCodeBench, and SuperGLUE are intentionally
excluded from the public aggregate because their upstream terms require local preparation or
additional redistribution permission.
