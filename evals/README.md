# Decision Engine Benchmark v1

Benchmark v1 evaluates the **routing policy**, not generated-answer quality.

It is intentionally selection-only: running it must not invoke a model or incur metered API spend. The benchmark separates two kinds of expectations so the results are not presented as stronger evidence than they are.

## Expectation types

### Hard expectations

These are behaviors that follow from explicit product constraints rather than subjective judgments about which model is "best":

- local-only and sensitive-data privacy enforcement;
- explicit cloud permission;
- explicitly declared reasoning or coding complexity;
- required capability support;
- context-window fit;
- explicit minimum-quality and request-budget constraints; and
- decline behavior when the service does not support a requested capability.

A violation of one of these expectations is a policy or implementation failure.

### Policy expectations

These test the product thesis rather than objective ground truth. Examples include whether an undeclared strategy prompt should be classified as moderate or high reasoning and whether the selected route is the cheapest adequate option.

A mismatch is evidence to review. It does **not** automatically mean the router is wrong: the expected policy, analyzer, model metadata, or benchmark case may need revision.

## What Benchmark v1 does not prove

Benchmark v1 does not compare the quality of generated answers. It therefore cannot prove that one model is objectively better than another, or that routing preserves answer quality while reducing cost.

Those claims require an execution benchmark with versioned model outputs, a quality rubric or trustworthy judge, repeated runs, latency and billed-cost capture, and explicit handling of judge disagreement.

## Dataset

`benchmark_v1.json` starts with 30 cases across:

- simple chat;
- inferred and explicit reasoning complexity;
- coding;
- privacy and cloud permission;
- unsupported capabilities;
- context-window constraints;
- request budget;
- latency preference; and
- explicit quality thresholds.

The first release is deliberately small enough to inspect manually. Additional cases should be added because they expose a meaningful boundary or failure mode, not to inflate the benchmark size.

## Review loop

The intended product loop is:

**benchmark → mismatch → classify failure → decide whether policy or expectation is wrong → change one thing → add regression case → rerun**

Do not tune the router simply to maximize an aggregate score.

## Next implementation step

The benchmark runner should:

1. load and validate every case;
2. construct a `TaskRequest`;
3. call `DecisionEngine.select` only;
4. capture routing latency, selected model, route metadata, inferred requirements, rejections, and fallbacks;
5. score hard expectations separately from policy expectations;
6. emit a machine-readable result artifact plus a concise Markdown report; and
7. return a non-zero exit code only for malformed benchmark data or configured hard-expectation regressions.

Live-readiness results should record the runtime snapshot because availability and quota can legitimately change between runs.
