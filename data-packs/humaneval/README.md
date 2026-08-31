# HumanEval data pack

This wheel contains a deterministic 100-record regression subset of OpenAI HumanEval at revision
`6d43fb980f9fee3c892a914eda09951f772ad10d`. Generated code must be evaluated only through the
remote executor; installing the pack never executes benchmark code.

The upstream benchmark and data are MIT licensed. This subset is labeled `regression_subset` and
must not be reported as the full official HumanEval score.
