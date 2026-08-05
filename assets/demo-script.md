# Self-hosted demo script

Record a real terminal session from the repository directory on the self-hosted Linux runtime. Start with a clean terminal, use a generic prompt, and review every frame for private data before converting the recording to `assets/demo.gif`.

These are the three selection-only commands used for the verified routing cases:

## 1. Simple task → Qwen

```bash
uv run ai-decision-engine "Explain this simply" --select-only --show-rejections
```

## 2. High reasoning → GPT-5.6 Sol

```bash
uv run ai-decision-engine "Evaluate three monetization strategies for an AI productivity platform. Compare risks, pricing power, and execution complexity, then recommend a staged launch strategy with explicit tradeoffs." --allow-cloud --select-only --show-rejections
```

## 3. Extreme reasoning → Claude Opus 4.8

```bash
uv run ai-decision-engine "Review the architecture and compare three deployment approaches. Assess privacy, security, reliability, and failure-mode risks; evaluate cost, latency, scalability, and operational tradeoffs; design a phased migration with rollback criteria; then synthesize the evidence and recommend an implementation strategy." --allow-cloud --select-only --show-rejections
```

The live session should show the engine's actual decision and rejection evidence, including Gemini exclusion when the current readiness signal reports rate limiting. Results can change when runtime readiness or quota changes; do not splice in expected output.

## Recording checklist

1. Confirm the configured model probes immediately before recording.
2. Hide the shell prompt and environment details or use a neutral prompt.
3. Run the commands without editing their output.
4. Review the recording frame by frame for secrets and private data.
5. Save the final recording as `assets/demo.gif`, then replace the README placeholder with the image.
