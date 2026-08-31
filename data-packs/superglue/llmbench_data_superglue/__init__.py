from __future__ import annotations

import json
from importlib.resources import files


def manifest() -> dict:
    return json.loads(files(__package__).joinpath("pack.json").read_text(encoding="utf-8"))
