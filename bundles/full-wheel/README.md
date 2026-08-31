# QuantTrio llmbench full wheel

This bundle produces one standards-compliant pure-Python wheel containing:

- the complete `quanttrio-llmbench` 1.0.1 framework and `llmbench` CLI;
- all core datasets and JSON schemas;
- all 13 publicly redistributable v0.5.0 data packs;
- the aggregate data entry point, source revisions, SHA256 metadata, and license notices.

The wheel tag is `py3-none-any`, which is the Python packaging standard for a pure-Python wheel
without extension modules. It is more portable than `abi3`: one file supports Python 3.10+
across macOS and Linux. An `abi3` tag would only be valid if this project shipped a CPython
limited-API binary extension and would require separate platform wheels.

Build after generating the consolidated public data pack:

```bash
python data-packs/all-public/build_pack.py
python bundles/full-wheel/build_bundle.py
python -m build --wheel bundles/full-wheel
```

Install the full wheel into a clean environment. The projects under `data-packs/` are internal
build inputs rather than separate public Release artifacts.
