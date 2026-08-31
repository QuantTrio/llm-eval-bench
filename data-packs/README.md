# Optional benchmark data wheels

Each subdirectory is an independently versioned Python project that exposes a
`llmbench.data_packs` entry point. Published packs contain only data whose redistribution terms
permit it. Restricted benchmarks use the same package format but must be built locally from an
official source archive.

Build and verify a pack:

```bash
python data-packs/humaneval/build_pack.py
python -m build data-packs/humaneval
python -m pip install data-packs/humaneval/dist/*.whl
llmbench data verify
```
