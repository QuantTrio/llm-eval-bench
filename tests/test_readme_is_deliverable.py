"""The minimal README remains usable from a checkout without optional data packs."""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest
from rich.text import Text
from typer.testing import CliRunner

from llmbench import data_packs
from llmbench.cli import app
from tests.test_provider_e2e import MODEL_ID, MockState, make_dataset, run_local_server

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"


@pytest.fixture(autouse=True)
def bare_clone(monkeypatch) -> None:
    monkeypatch.setattr(data_packs, "_entry_points", list)
    for key in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "XAI_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "CHAT_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)


def readme_commands() -> list[list[str]]:
    blocks = re.findall(r"```bash\n(.*?)```", README.read_text(encoding="utf-8"), re.DOTALL)
    joined = "\n".join(blocks).replace("\\\n", " ")
    return [
        shlex.split(line)[1:]
        for line in joined.splitlines()
        if line.strip().startswith("llmbench ")
    ]


def test_readme_commands_and_options_exist() -> None:
    runner = CliRunner()
    commands = readme_commands()
    assert commands
    for command in commands:
        result = runner.invoke(app, [command[0], "--help"])
        assert result.exit_code == 0, result.output
        visible = Text.from_ansi(result.output).plain
        for option in (part for part in command if part.startswith("--")):
            assert option in visible, f"README option {option} is unavailable: {command}"
        if command[0] in {"run", "eval", "stress"} and "--resume" not in command:
            assert "--model" in command or "--config" in command, command


def test_readme_usage_runs_without_optional_services(tmp_path, monkeypatch) -> None:
    """Execute the actual quick-start examples with loopback URLs and a bounded sample."""
    monkeypatch.chdir(tmp_path)
    dataset = tmp_path / "custom.jsonl"
    make_dataset(dataset)
    state = MockState()
    runner = CliRunner()
    run_paths = []
    first_run_command = None
    with run_local_server(state) as base:
        for index, original in enumerate(readme_commands()):
            command = [MODEL_ID if arg == "MODEL" else arg for arg in original]
            if "--model" in command:
                command[command.index("--model") + 1] = MODEL_ID
            if command[0] == "run" and "--resume" not in command:
                if "--base-url" in command:
                    command[command.index("--base-url") + 1] = f"{base}/v1"
                else:
                    command.extend(["--base-url", f"{base}/v1"])
                # Default/preset data is checked separately; fixture data makes the score known.
                output = tmp_path / f"readme-{index}"
                command.extend(
                    [
                        "--dataset",
                        str(dataset),
                        "--limit",
                        "1",
                        "--mode",
                        "quality",
                        "--output-dir",
                        str(output),
                    ]
                )
                run_paths.append(output)
                first_run_command = first_run_command or command
            elif command[0] == "run" and "--resume" in command:
                assert run_paths
                command[command.index("--resume") + 1] = str(run_paths[0])
            elif command[0] == "compare":
                # The README comparison requires two existing runs of the same protocol.
                assert run_paths
                assert first_run_command
                candidate = tmp_path / "comparison-candidate"
                second = runner.invoke(app, [*first_run_command, "--output-dir", str(candidate)])
                assert second.exit_code == 0, (second.output, second.exception)
                command[command.index("--baseline") + 1] = str(run_paths[0])
                command[command.index("--candidate") + 1] = str(candidate)
                command.extend(["--report", str(tmp_path / "comparison.html")])
            result = runner.invoke(app, command)
            assert result.exit_code == 0, (command, result.output, result.exception)
    assert run_paths
    for directory in run_paths:
        for name in ("summary.json", "raw_results.jsonl", "events.jsonl", "report.html"):
            assert (directory / name).is_file()
    assert state.calls and all(call["method"] == "POST" for call in state.calls)
    assert (tmp_path / "comparison.html").is_file()


def test_yaml_example_supplies_model_id(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    source = (REPO / "examples/bench.yaml").read_text(encoding="utf-8")
    state = MockState()
    with run_local_server(state) as base:
        config = tmp_path / "bench.yaml"
        config.write_text(
            source.replace("http://127.0.0.1:8000/v1", f"{base}/v1").replace(
                "model: local-model", f"model: {MODEL_ID}"
            ),
            encoding="utf-8",
        )
        result = CliRunner().invoke(
            app,
            [
                "run",
                "--config",
                str(config),
                "--mode",
                "quality",
                "--limit",
                "1",
                "--output-dir",
                str(tmp_path / "configured"),
            ],
        )
    assert result.exit_code == 0, (result.output, result.exception)
    assert state.calls and all(call["method"] == "POST" for call in state.calls)
    assert all(call["body"].get("model") == MODEL_ID for call in state.calls)


def test_readme_default_datasets_are_bundled() -> None:
    from llmbench.datasets import read_manifest
    from llmbench.runspec import DEFAULT_DATASETS

    assert set(DEFAULT_DATASETS) <= set(read_manifest())


def test_readme_links_resolve_in_checkout() -> None:
    for target in re.findall(r"\]\(([^)]+)\)", README.read_text(encoding="utf-8")):
        if "://" not in target and not target.startswith("#"):
            assert (REPO / target.split("#")[0]).is_file(), target
