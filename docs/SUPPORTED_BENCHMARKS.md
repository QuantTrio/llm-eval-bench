# Stable representative benchmark matrix

The stable suite supports 20 DataLearner categories. HLE (`综合评估`) and Fiction.liveBench
(`长上下文`) remain visible in the curated 155-entry catalog snapshot but are intentionally not
part of the executable/stable coverage contract.

| Category | Representative | Delivery | Capability |
|---|---|---|---|
| AI Agent - 信息收集 | BrowseComp | encrypted wheel | agent |
| AI Agent - 工具使用 | Terminal-Bench 2.0 | descriptor wheel | agent |
| Agent能力评测 | Aider Polyglot | descriptor wheel | agent |
| OpenClaw智能体能力综合测评 | PinchBench | descriptor wheel | agent |
| 代码能力 | HumanEval | wheel | agent |
| 写作和创作 | Creative Writing v3 | local-build wheel | chat + Judge |
| 图像向量嵌入 | MMEB-v2 Image mini | local-build wheel | embedding |
| 多模态理解 | MMMU | large wheel | multimodal |
| 常识推理 | SimpleBench public | wheel | chat |
| 常识问答 | SimpleQA | wheel | chat + Judge |
| 指令跟随 | IFBench | wheel | official verifier |
| 数学推理 | AIME 2025 | wheel | chat |
| 生产力知识 | GDPval gold | local-build descriptor | agent + Judge |
| 真实性评估 | TruthfulQA | core | chat |
| 科学与综合推理 | GPQA Diamond | core | chat |
| 综合能力 | AGIEval | wheel | chat |
| 编程与软件工程 | LiveCodeBench | local-build wheel | agent |
| 自然语言理解 | SuperGLUE | local-build wheel | chat |
| 长上下文能力 | LongBench v2 | large wheel | chat |
| 阅读理解 | DROP | core | chat |

Run `llmbench coverage` to inspect installation readiness and `llmbench data verify` to validate
every installed wheel's count and SHA256.
