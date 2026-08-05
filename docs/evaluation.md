# Evaluation

AI Decision Engine v0.1.0 has three live-validated routing cases on a self-hosted Linux runtime. They verify routing decisions against the configured OpenClaw catalog and readiness probes; they are not model-quality benchmarks and no generated answers are presented as evidence.

## Live routing cases

| Case | Request characteristics | Verified selection | Key rejection or ranking reasons |
|---|---|---|---|
| Simple task | Short text; local-only default; low reasoning | `ollama/qwen3:8b` | Cloud candidates are blocked without explicit cloud permission; higher-capability routes are unnecessary. |
| High reasoning | Multi-option strategy analysis; cloud explicitly allowed | `openai/gpt-5.6-sol` | Qwen is below the high-reasoning adequacy threshold. Opus remains an adequate fallback but GPT-5.6 Sol is primary for the configured high-reasoning tier. |
| Extreme reasoning | Broad architecture, risk, cost, reliability, migration, and trade-off analysis; cloud explicitly allowed | `anthropic/claude-opus-4-8` | Qwen is below the extreme-reasoning adequacy threshold. GPT-5.6 Sol is the first fallback; Opus is primary for the configured extreme-reasoning tier. |

Across the validated readiness snapshot, Claude Sonnet 5 is excluded when its model probe has not succeeded. Gemini 3.1 Pro Preview is excluded when the readiness signal reports rate limiting or unavailable quota. Catalog presence alone never makes a model eligible.

The exact selection-only commands are maintained in [`assets/demo-script.md`](../assets/demo-script.md). Their output must be recorded from the live host; it is intentionally not reproduced here.

## What the automated suite covers

The evaluation package contains typed synthetic cases for chat, coding, reasoning, privacy, unsupported current information/images, availability, and budget behavior. Tests also cover candidate scoring, fallback order, catalog failures, readiness mapping, privacy denial, quota rejection, and context enforcement.

## Interpretation and limitations

- A successful routing case proves the configured policy selected the expected eligible model at that time.
- Readiness and quota are runtime facts and may change between runs.
- Confidence describes task-analysis evidence and adequacy margin; it is not a probability that an answer is correct.
- Quality and speed scores remain provisional until versioned prompts, runtime/model versions, latency, cost, and human or trustworthy quality judgments are collected over repeated runs.
- Multimodal cases remain expected failures until image input is supported end to end.

No evaluation artifact should contain credentials, identity data, network addresses, or machine-specific paths.
