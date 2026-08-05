# Changelog

## 0.1.0 - 2026-08-04

First public, early-stage release. The routing engine has been live-validated on a self-hosted Linux runtime, but this version does not claim production maturity.

- Added a vendor-neutral cheapest-adequate routing pipeline, conservative Ollama text execution/discovery, local-only privacy defaults, transactional SQLite budget reservations, charge-aware fallback accounting, validation, CLI, evaluation schemas, tests, documentation, and CI.
- Added selection-only routing, canonical model references, billing/quota modes, cached OpenClaw CLI catalog discovery, and authenticated OpenResponses Gateway execution without direct cloud-provider clients.
- Aligned catalog readiness with the self-hosted OpenClaw 2026.7.1-2 runtime and restricted the approved pool to readiness-verified routes.
- Replaced capability keyword routing with typed reasoning, coding, multimodal, and latency requirements; added per-capability adequacy, routing confidence, and complete explainable reports.
- Added v1 model assessments for reasoning, coding, context, privacy, availability, quota, speed, and cost; all Decision Engine generation now crosses the OpenClaw execution boundary.
- Expanded the deny-by-default approved model pool with verified Opus and configured-but-gated Sonnet/Gemini routes, plus an extreme-reasoning adequacy tier and 2026.7.1-2 readiness fixtures.
- Added concise first-release documentation, verified demo instructions, evaluation notes, and a dry-run-capable self-hosted deployment helper.
