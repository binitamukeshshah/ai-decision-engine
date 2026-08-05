# Architecture decisions

## Local first is a policy, not a hardcoded provider

Local models receive a score preference only after satisfying hard quality and capability requirements. This prevents cheap but inadequate routing.

## Availability is runtime state

Definitions describe approved models; `/api/tags` determines whether Ollama models are installed. An unreachable provider produces unknown—not falsely unavailable—status.

## SQLite reservation ledger

SQLite provides durable transactions and cross-process `BEGIN IMMEDIATE` serialization without adding a service dependency. Integer nanodollar accounting preserves exact budget boundaries. Cloud routing fails closed when integrity or I/O errors make the ledger untrustworthy.

## No pretend cloud support

Cloud provider enum values and pricing fields establish extension points. No adapter is registered until real credentials, execution, and tests exist.

## Heuristic analyzer behind a protocol

Regex heuristics are deterministic and testable. The interface permits a learned/model-assisted classifier later without changing routing.

## Standalone product repository

`ai-decision-engine` should be a standalone repository. `ai-lab` should remain an umbrella portfolio repository containing architecture, roadmap, and links to substantial products such as this engine, Forge, Telegram Assistant, and Interview Coach.

## OpenClaw owns cloud execution

The router does not duplicate OAuth or subscription authentication. OpenClaw owns credentials, quotas, sessions, and all generation execution; the router supplies a canonical model decision. The direct Ollama adapter remains available for infrastructure-level discovery and integration testing but is not called by the Decision Engine for generation.
