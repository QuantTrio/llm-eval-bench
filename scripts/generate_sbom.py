#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import tomllib


def dependency_name(requirement: str) -> str:
    return re.split(r"[<>=!~;\[]", requirement, maxsplit=1)[0].strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("artifacts", nargs="*", type=Path)
    args = parser.parse_args()
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]
    components = []
    for requirement in project.get("dependencies", []):
        name = dependency_name(requirement)
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = "not-installed"
        components.append(
            {
                "type": "library",
                "name": name,
                "version": version,
                "purl": f"pkg:pypi/{name}@{version}",
            }
        )
    artifact_components = []
    for path in args.artifacts:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        artifact_components.append(
            {
                "type": "file",
                "name": path.name,
                "hashes": [{"alg": "SHA-256", "content": digest}],
            }
        )
    payload = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:llmbench-{project['version']}",
        "version": 1,
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "component": {
                "type": "application",
                "name": project["name"],
                "version": project["version"],
            },
        },
        "components": components + artifact_components,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote SBOM to {args.output}")


if __name__ == "__main__":
    main()
