from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from .schemas import RequestResult


def _fmt(value: Any, *, percent: bool = False) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value * 100:.2f}%" if percent else f"{value:.2f}"
    return str(value)


def markdown_report(summary: dict[str, Any], *, title: str = "Model Benchmark Report") -> str:
    quality = summary["quality"]
    performance = summary["performance"]
    config = summary["config"]
    sample_score = quality.get("sample_mean_score", quality.get("mean_score"))
    lines = [
        f"# {title}",
        "",
        f"- Run ID: `{summary['run_id']}`",
        f"- Mode: `{summary['mode']}`",
        f"- Model: `{summary['model']}`",
        f"- Base URL: `{summary['base_url']}`",
        f"- Datasets: `{', '.join(config.get('datasets', [])) or 'stress'}`",
        f"- Concurrency: `{config['concurrency']}`",
        "",
        "## Quality",
        "",
        f"- Sample mean score (mixed metrics): {_fmt(sample_score, percent=True)}",
        f"- Composite score: {_fmt(quality.get('composite_score'), percent=True)}",
        f"- Quality valid: `{quality.get('quality_valid', True)}`",
        f"- Parse fail rate: {_fmt(quality.get('parse_fail_rate'), percent=True)}",
    ]
    if quality.get("pass_at_k") is not None:
        lines.extend(
            [
                f"- Accuracy@1: {_fmt(quality.get('accuracy_at_1'), percent=True)}",
                f"- Pass@{quality.get('pass_k')}: {_fmt(quality.get('pass_at_k'), percent=True)}",
                f"- Majority@{quality.get('pass_k')}: "
                f"{_fmt(quality.get('majority_at_k'), percent=True)}",
                f"- Consistency@{quality.get('pass_k')}: "
                f"{_fmt(quality.get('consistency_at_k'), percent=True)}",
            ]
        )
    lines.extend(
        [
            "",
            "| Dataset | Metric | Samples | Questions | Score | Parse fail | Truncated | Valid |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for dataset, values in quality.get("by_dataset", {}).items():
        lines.append(
            f"| {dataset} | {values['metric']} | {values['samples']} | {values['questions']} | "
            f"{_fmt(values.get('score', values.get('mean_score')), percent=True)} | "
            f"{_fmt(values['parse_fail_rate'], percent=True)} | "
            f"{_fmt(values.get('truncation_rate'), percent=True)} | "
            f"{values.get('quality_valid', True)} |"
        )
    lines.extend(
        [
            "",
            "### Category breakdown",
            "",
            "| Category | Datasets | Samples | Macro score |",
            "|---|---|---:|---:|",
        ]
    )
    for category, values in quality.get("by_category", {}).items():
        lines.append(
            f"| {category} | {', '.join(values['datasets'])} | {values['samples']} | "
            f"{_fmt(values['macro_mean_score'], percent=True)} |"
        )
    lines.extend(
        [
            "",
            "### Question-type breakdown",
            "",
            "| Type | Samples | Mean score |",
            "|---|---:|---:|",
        ]
    )
    for question_type, values in quality.get("by_question_type", {}).items():
        lines.append(
            f"| {question_type} | {values['samples']} | "
            f"{_fmt(values.get('score', values.get('mean_score')), percent=True)} |"
        )
    p95_latency = performance["latency_ms"].get("p95")
    lines.extend(
        [
            "",
            "## Performance",
            "",
            f"- Total requests: {performance['total_requests']}",
            f"- Successful requests: {performance['successful_requests']}",
            f"- Unsupported requests: {performance.get('unsupported_requests', 0)}",
            f"- Failed requests: {performance['failed_requests']}",
            f"- QPS: {_fmt(performance['qps'])}",
            f"- Output tokens/s: {_fmt(performance['output_tokens_per_second'])}",
            f"- Latency p95: {_fmt(p95_latency)} ms",
            f"- TTFT p95: {_fmt(performance['ttft_ms'].get('p95'))} ms",
            f"- TPOT p95: {_fmt(performance['tpot_ms'].get('p95'))} ms",
            f"- Error rate: {_fmt(performance['error_rate'], percent=True)}",
            f"- Timeout rate: {_fmt(performance['timeout_rate'], percent=True)}",
            f"- Truncated responses: {performance['truncated_responses']}",
            f"- Truncation rate: {_fmt(performance.get('truncation_rate'), percent=True)}",
            "",
            "## Errors",
            "",
        ]
    )
    if performance["errors"]:
        lines.extend(f"- `{kind}`: {count}" for kind, count in performance["errors"].items())
    else:
        lines.append("No request errors.")
    lines.append("")
    return "\n".join(lines)


def html_report(
    summary: dict[str, Any],
    results: list[RequestResult] | None = None,
    *,
    title: str = "Model Benchmark Report",
) -> str:
    markdown = markdown_report(summary, title=title)
    body = "\n".join(
        f"<p>{html.escape(line)}</p>" if line else "" for line in markdown.splitlines()
    )
    quality = summary["quality"]
    performance = summary["performance"]
    cards = [
        (
            "Sample mean",
            _fmt(quality.get("sample_mean_score", quality.get("mean_score")), percent=True),
        ),
        ("QPS", _fmt(performance.get("qps"))),
        ("Latency p95", f"{_fmt(performance['latency_ms'].get('p95'))} ms"),
        ("Error rate", _fmt(performance.get("error_rate"), percent=True)),
    ]
    card_html = "".join(
        f'<div class="card"><span>{html.escape(label)}</span>'
        f"<strong>{html.escape(value)}</strong></div>"
        for label, value in cards
    )
    dataset_rows = []
    for dataset, values in quality.get("by_dataset", {}).items():
        score = values.get("score", values.get("mean_score"))
        width = max(0.0, min(100.0, float(score or 0) * 100))
        dataset_rows.append(
            f"<tr><td>{html.escape(dataset)}</td><td>{html.escape(values['metric'])}</td>"
            f"<td>{_fmt(score, percent=True)}</td><td><div class='bar' style='width:{width:.2f}%'>"
            "</div></td>"
            f"<td>{_fmt(values.get('truncation_rate'), percent=True)}</td>"
            f"<td>{values.get('quality_valid', True)}</td></tr>"
        )
    details = []
    for result in results or []:
        status = result.error_type or ("truncated" if result.finish_reason == "length" else "ok")
        details.append(
            "<tr>"
            f"<td>{html.escape(result.dataset)}</td>"
            f"<td>{html.escape(result.question_id)}</td>"
            f"<td>{_fmt(result.score, percent=True)}</td>"
            f"<td>{html.escape(status)}</td>"
            "<td><details><summary>view</summary>"
            f"<strong>Prompt</strong><pre>{html.escape(result.prompt)}</pre>"
            f"<strong>Output</strong><pre>{html.escape(result.raw_output)}</pre>"
            f"<strong>Parsed</strong><pre>{html.escape(str(result.parsed_answer))}</pre>"
            "</details></td></tr>"
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>{html.escape(title)}</title>
<style>
:root{{--bg:#0b1020;--panel:#141b2d;--text:#e8ecf5;--muted:#9aa7bd;--accent:#65d6c4}}
body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 system-ui,sans-serif}}
main{{max-width:980px;margin:48px auto;padding:0 24px}} h1{{font-size:34px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
gap:12px;margin:24px 0}}
.card{{background:var(--panel);border:1px solid #25314d;border-radius:12px;padding:18px}}
.card span{{display:block;color:var(--muted)}} .card strong{{font-size:25px;color:var(--accent)}}
pre{{white-space:pre-wrap;background:var(--panel);padding:24px;border-radius:12px;overflow:auto}}
table{{width:100%;border-collapse:collapse;margin:18px 0}}
th,td{{padding:8px;border-bottom:1px solid #25314d;text-align:left}}
.bar{{height:12px;background:var(--accent);border-radius:6px;min-width:2px}}
details pre{{max-height:260px}}
a{{color:var(--accent)}}
</style></head><body><main><h1>{html.escape(title)}</h1><div class="cards">{card_html}</div>
<h2>Dataset scores</h2><table><thead><tr><th>Dataset</th><th>Metric</th>
<th>Score</th><th>Bar</th><th>Truncated</th><th>Valid</th></tr></thead>
<tbody>{"".join(dataset_rows)}</tbody></table>
<h2>Question details</h2><table><thead><tr><th>Dataset</th><th>Question</th>
<th>Score</th><th>Status</th><th>Details</th></tr></thead>
<tbody>{"".join(details)}</tbody></table>
<pre>{html.escape(markdown)}</pre><details><summary>Accessible paragraph view</summary>
{body}</details>
</main></body></html>"""


def write_run_artifacts(
    output_dir: Path,
    summary: dict[str, Any],
    results: list[RequestResult],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / "summary.json",
        "raw": output_dir / "raw_results.jsonl",
        "markdown": output_dir / "report.md",
        "html": output_dir / "report.html",
    }
    paths["summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with paths["raw"].open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
    paths["markdown"].write_text(markdown_report(summary), encoding="utf-8")
    paths["html"].write_text(html_report(summary, results), encoding="utf-8")
    return paths


def _relative_change(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline in {None, 0}:
        return None
    return (candidate - baseline) / baseline


def compare_summaries(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    base_quality = baseline["quality"]
    cand_quality = candidate["quality"]
    base_overall = base_quality.get(
        "composite_score", base_quality.get("sample_mean_score", base_quality.get("mean_score"))
    )
    cand_overall = cand_quality.get(
        "composite_score", cand_quality.get("sample_mean_score", cand_quality.get("mean_score"))
    )
    if base_overall is None:
        base_overall = base_quality.get("sample_mean_score")
    if cand_overall is None:
        cand_overall = cand_quality.get("sample_mean_score")
    datasets: dict[str, Any] = {}
    names = sorted(
        set(base_quality.get("by_dataset", {})) | set(cand_quality.get("by_dataset", {}))
    )
    for name in names:
        base_values = base_quality.get("by_dataset", {}).get(name) or {}
        cand_values = cand_quality.get("by_dataset", {}).get(name) or {}
        base_score = base_values.get(
            "score", base_values.get("mean_score", base_values.get("accuracy"))
        )
        cand_score = cand_values.get(
            "score", cand_values.get("mean_score", cand_values.get("accuracy"))
        )
        datasets[name] = {
            "baseline": base_score,
            "candidate": cand_score,
            "absolute_change": (
                cand_score - base_score
                if base_score is not None and cand_score is not None
                else None
            ),
            "relative_change": _relative_change(cand_score, base_score),
        }
    base_perf = baseline["performance"]
    cand_perf = candidate["performance"]
    base_p95 = base_perf["latency_ms"].get("p95")
    cand_p95 = cand_perf["latency_ms"].get("p95")
    base_memory = baseline.get("config", {}).get("memory_gb")
    cand_memory = candidate.get("config", {}).get("memory_gb")
    return {
        "schema_version": 1,
        "baseline": {"run_id": baseline["run_id"], "model": baseline["model"]},
        "candidate": {"run_id": candidate["run_id"], "model": candidate["model"]},
        "quality": {
            "baseline_score": base_overall,
            "candidate_score": cand_overall,
            "absolute_change": (
                cand_overall - base_overall
                if base_overall is not None and cand_overall is not None
                else None
            ),
            "relative_change": _relative_change(cand_overall, base_overall),
            "by_dataset": datasets,
        },
        "performance": {
            "throughput_change": _relative_change(
                cand_perf.get("output_tokens_per_second"),
                base_perf.get("output_tokens_per_second"),
            ),
            "qps_change": _relative_change(cand_perf.get("qps"), base_perf.get("qps")),
            "p95_latency_reduction": (
                (base_p95 - cand_p95) / base_p95
                if base_p95 not in {None, 0} and cand_p95 is not None
                else None
            ),
            "error_rate_change": cand_perf.get("error_rate", 0) - base_perf.get("error_rate", 0),
            "memory_reduction": (
                (base_memory - cand_memory) / base_memory
                if base_memory not in {None, 0} and cand_memory is not None
                else None
            ),
        },
    }


def comparison_markdown(comparison: dict[str, Any]) -> str:
    quality = comparison["quality"]
    performance = comparison["performance"]
    overall_change = quality.get("sample_mean_change", quality.get("absolute_change"))
    confidence = quality.get("confidence_interval") or {}
    p95_change = performance.get("p95_latency_change", performance.get("p95_latency_reduction"))
    lines = [
        "# Quantization Regression Report",
        "",
        f"Baseline: `{comparison['baseline']['model']}` (`{comparison['baseline']['run_id']}`)",
        f"Candidate: `{comparison['candidate']['model']}` (`{comparison['candidate']['run_id']}`)",
        "",
        f"Sample mean change: {_fmt(overall_change, percent=True)}",
        f"95% paired CI: {_fmt(confidence.get('lower'), percent=True)} to "
        f"{_fmt(confidence.get('upper'), percent=True)}",
        f"Throughput change: {_fmt(performance['throughput_change'], percent=True)}",
        f"QPS change: {_fmt(performance['qps_change'], percent=True)}",
        f"p95 latency change: {_fmt(p95_change, percent=True)}",
        f"Error-rate change: {_fmt(performance['error_rate_change'], percent=True)}",
        f"Memory reduction: {_fmt(performance.get('memory_reduction'), percent=True)}",
        "",
        "| Dataset | Baseline | Candidate | Absolute | Relative |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in quality["by_dataset"].items():
        lines.append(
            f"| {name} | {_fmt(values['baseline'], percent=True)} | "
            f"{_fmt(values['candidate'], percent=True)} | "
            f"{_fmt(values['absolute_change'], percent=True)} | "
            f"{_fmt(values['relative_change'], percent=True)} |"
        )
    lines.append("")
    if comparison.get("policy"):
        lines.extend(["## Regression policy", ""])
        for gate in comparison["policy"]["gates"]:
            marker = "PASS" if gate["passed"] else "FAIL"
            lines.append(
                f"- **{marker}** `{gate['gate']}`: actual={_fmt(gate['actual'])}, "
                f"limit={_fmt(gate['limit'])}"
            )
        lines.append("")
    return "\n".join(lines)


def write_comparison(output: Path, comparison: dict[str, Any]) -> dict[str, Path]:
    if output.suffix:
        directory = output.parent
        html_path = output
    else:
        directory = output
        html_path = output / "compare.html"
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "compare.json"
    md_path = directory / "compare.md"
    paired_path = directory / "paired_results.jsonl"
    markdown = comparison_markdown(comparison)
    summary_payload = {key: value for key, value in comparison.items() if key != "paired_results"}
    json_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(markdown, encoding="utf-8")
    if comparison.get("paired_results") is not None:
        with paired_path.open("w", encoding="utf-8") as handle:
            for result in comparison["paired_results"]:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    quality = comparison["quality"]
    performance = comparison["performance"]
    cards = "".join(
        f'<div class="card"><span>{html.escape(label)}</span>'
        f"<strong>{html.escape(value)}</strong></div>"
        for label, value in [
            ("Baseline", comparison["baseline"]["model"]),
            ("Candidate", comparison["candidate"]["model"]),
            (
                "Score change",
                _fmt(
                    quality.get("sample_mean_change", quality.get("absolute_change")),
                    percent=True,
                ),
            ),
            (
                "Throughput change",
                _fmt(performance["throughput_change"], percent=True),
            ),
        ]
    )
    paired_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['dataset']))}</td>"
        f"<td>{html.escape(str(row['question_id']))}</td>"
        f"<td>{html.escape(str(row['transition']))}</td>"
        f"<td>{_fmt(row.get('baseline_score'), percent=True)}</td>"
        f"<td>{_fmt(row.get('candidate_score'), percent=True)}</td>"
        f"<td>{html.escape(str(row.get('baseline_answer')))}</td>"
        f"<td>{html.escape(str(row.get('candidate_answer')))}</td>"
        "</tr>"
        for row in comparison.get("paired_results", [])
    )
    html_document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width"><title>Quantization Regression Report</title>
<style>body{{font:15px/1.55 system-ui,sans-serif;max-width:980px;margin:48px auto;padding:0 24px;
background:#0b1020;color:#e8ecf5}}.cards{{display:grid;grid-template-columns:repeat(2,1fr);
gap:12px}}.card,pre{{background:#141b2d;border:1px solid #25314d;border-radius:12px;padding:18px}}
.card span{{display:block;color:#9aa7bd}}.card strong{{font-size:20px;color:#65d6c4}}
pre{{white-space:pre-wrap;overflow:auto}}table{{width:100%;border-collapse:collapse;margin:20px 0}}
th,td{{padding:7px;border-bottom:1px solid #25314d;text-align:left}}</style></head><body>
<h1>Quantization Regression Report</h1>
<div class="cards">{cards}</div><h2>Paired question changes</h2>
<table><thead><tr><th>Dataset</th><th>Question</th><th>Transition</th><th>Baseline</th>
<th>Candidate</th><th>Baseline answer</th><th>Candidate answer</th></tr></thead>
<tbody>{paired_rows}</tbody></table><pre>{html.escape(markdown)}</pre></body></html>"""
    html_path.write_text(html_document, encoding="utf-8")
    paths = {"json": json_path, "markdown": md_path, "html": html_path}
    if comparison.get("paired_results") is not None:
        paths["paired"] = paired_path
    return paths


def write_sweep_artifacts(output_dir: Path, payload: dict[str, Any]) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "sweep.json"
    markdown_path = output_dir / "sweep.md"
    html_path = output_dir / "sweep.html"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Concurrency Sweep",
        "",
        f"- Model: `{payload['model']}`",
        f"- Prompt profile: `{payload['prompt_profile']}`",
        "",
        "| Concurrency | Requests | QPS | Output tokens/s | "
        "TTFT p95 ms | Latency p95 ms | Errors |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for point in payload["points"]:
        performance = point["summary"]["performance"]
        lines.append(
            f"| {point['concurrency']} | {performance['total_requests']} | "
            f"{performance['qps']:.2f} | {performance['output_tokens_per_second']:.2f} | "
            f"{_fmt(performance['ttft_ms']['p95'])} | "
            f"{_fmt(performance['latency_ms']['p95'])} | "
            f"{performance['failed_requests']} |"
        )
    markdown = "\n".join(lines) + "\n"
    markdown_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Concurrency Sweep</title>"
        "<style>body{font:15px/1.5 system-ui;max-width:1000px;margin:40px auto;padding:0 20px}"
        "pre{white-space:pre-wrap}</style></head><body><pre>"
        + html.escape(markdown)
        + "</pre></body></html>",
        encoding="utf-8",
    )
    return {"json": json_path, "markdown": markdown_path, "html": html_path}
