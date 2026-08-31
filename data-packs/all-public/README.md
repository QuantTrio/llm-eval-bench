# Internal consolidated data build input

This project stages the 13 publicly redistributable benchmark assets used to build
`quanttrio-llmbench-full`. It is maintained as an internal build input and is not published as a
separate Release artifact.

The staging package provides AGIEval, Aider Polyglot, AIME 2025, BrowseComp, HumanEval, IFBench,
LongBench v2, MMMU, MTEB retrieval mini, PinchBench, SimpleBench, SimpleQA, and Terminal-Bench 2.0.
Together with the three representative datasets in the core wheel, coverage is `16/21`.

Maintainers rebuild the staging package after generating each source pack:

```bash
python data-packs/all-public/build_pack.py
python -m build data-packs/all-public
```

Creative Writing v3, MMEB-v2 Image, GDPval Gold, LiveCodeBench, and SuperGLUE are excluded because
their upstream terms require local preparation or additional redistribution permission.
