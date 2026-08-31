# Optional benchmark data wheels

Each subdirectory is an independently versioned Python project that exposes a
`llmbench.data_packs` entry point. Published packs contain only data whose redistribution terms
permit it. Restricted benchmarks use the same package format but must be built locally from an
official source archive.

For the simplest installation, `all-public/` builds one real wheel containing all 13 publicly
redistributable packs. Install that aggregate wheel by itself, or install selected individual
wheels; do not mix both delivery forms because they provide the same dataset IDs.

Build and verify a pack:

```bash
python data-packs/humaneval/build_pack.py
python -m build data-packs/humaneval
python -m pip install data-packs/humaneval/dist/*.whl
llmbench data verify
```

Build the public aggregate after its 13 source packs have been generated:

```bash
python data-packs/all-public/build_pack.py
python -m build data-packs/all-public
```
