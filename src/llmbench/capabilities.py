from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class BenchmarkCapability:
    category: str
    benchmark: str
    dataset_id: str
    capability: str
    adapter: str
    default_size: int
    data_pack: str | None = None
    recommended_max_tokens: int = 4096


BENCHMARK_CAPABILITIES = (
    BenchmarkCapability(
        "AI Agent - 信息收集",
        "BrowseComp",
        "browsecomp",
        "agent",
        "remote_browser",
        100,
        "browsecomp",
    ),
    BenchmarkCapability(
        "AI Agent - 工具使用",
        "Terminal-Bench 2.0",
        "terminal-bench-2",
        "agent",
        "remote_executor",
        89,
        "terminal-bench-2",
    ),
    BenchmarkCapability(
        "Agent能力评测",
        "Aider Polyglot",
        "aider-polyglot",
        "agent",
        "remote_executor",
        200,
        "aider-polyglot",
    ),
    BenchmarkCapability(
        "OpenClaw智能体能力综合测评",
        "PinchBench",
        "pinchbench",
        "agent",
        "remote_executor",
        100,
        "pinchbench",
    ),
    BenchmarkCapability(
        "代码能力", "HumanEval", "humaneval", "agent", "remote_executor", 100, "humaneval"
    ),
    BenchmarkCapability(
        "写作和创作",
        "Creative Writing v3",
        "creative-writing-v3",
        "chat",
        "judge",
        96,
        "creative-writing-v3",
    ),
    BenchmarkCapability(
        "图像向量嵌入",
        "MMEB-v2 Image",
        "mmeb-v2-image",
        "embedding",
        "embedding",
        100,
        "mmeb-v2-image",
    ),
    BenchmarkCapability("多模态理解", "MMMU", "mmmu", "multimodal", "multimodal_chat", 500, "mmmu"),
    BenchmarkCapability(
        "常识推理", "SimpleBench", "simple-bench", "chat", "native", 10, "simple-bench"
    ),
    BenchmarkCapability("常识问答", "SimpleQA", "simpleqa", "chat", "judge", 500, "simpleqa"),
    BenchmarkCapability(
        "指令跟随", "IFBench", "ifbench", "chat", "official_verifier", 200, "ifbench"
    ),
    BenchmarkCapability(
        "数学推理",
        "AIME 2025",
        "aime-2025",
        "chat",
        "native",
        30,
        "aime-2025",
        8192,
    ),
    BenchmarkCapability(
        "文本向量检索",
        "MTEB retrieval mini",
        "mteb-retrieval-mini",
        "embedding",
        "embedding",
        500,
        "mteb-retrieval-mini",
    ),
    BenchmarkCapability(
        "生产力知识",
        "GDPval gold",
        "gdpval-gold",
        "agent",
        "artifact_judge",
        200,
        "gdpval-gold",
        8192,
    ),
    BenchmarkCapability("真实性评估", "TruthfulQA", "truthfulqa", "chat", "native", 200),
    BenchmarkCapability("科学与综合推理", "GPQA Diamond", "gpqa-diamond", "chat", "native", 100),
    BenchmarkCapability("综合能力", "AGIEval", "agieval", "chat", "native", 200, "agieval"),
    BenchmarkCapability("综合评估", "HLE", "hle", "multimodal", "judge", 100, "hle", 8192),
    BenchmarkCapability(
        "编程与软件工程",
        "LiveCodeBench",
        "livecodebench",
        "agent",
        "remote_executor",
        100,
        "livecodebench",
    ),
    BenchmarkCapability(
        "自然语言理解", "SuperGLUE", "superglue", "chat", "native", 500, "superglue"
    ),
    BenchmarkCapability(
        "长上下文",
        "Fiction.liveBench",
        "fiction-livebench",
        "chat",
        "judge",
        36,
        "fiction-livebench",
        8192,
    ),
    BenchmarkCapability(
        "长上下文能力", "LongBench v2", "longbench-v2", "chat", "judge", 200, "longbench-v2", 8192
    ),
    BenchmarkCapability("阅读理解", "DROP", "drop", "chat", "native", 500),
)


def capability_matrix(installed_datasets: set[str]) -> list[dict]:
    rows = []
    for item in BENCHMARK_CAPABILITIES:
        row = asdict(item)
        row["installed"] = item.dataset_id in installed_datasets
        rows.append(row)
    return rows
