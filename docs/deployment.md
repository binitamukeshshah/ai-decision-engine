# Deployment target

## MacBook

Use for development, Codex, tests, Git, and GitHub. Direct Ollama can be reached through an SSH tunnel when desired.

## Self-hosted Linux runtime

Run OpenClaw Gateway, the AI Decision Engine service, Ollama, and Telegram runtime. Enable OpenClaw's Responses HTTP endpoint, retain Gateway authentication, bind privately, and supply the operator token through approved secret/environment configuration.

Verify the installed OpenClaw release before deployment:

```bash
openclaw models list --provider ollama --json
openclaw models list --provider openai --json
openclaw models list --provider anthropic --json
openclaw models list --provider google --json
openclaw models status --probe --json
```

The combined `openclaw models list --json` command is not used by default because it fails on the observed `2026.7.1-2` installation. Configure the scoped discovery set with `ADE_OPENCLAW_CATALOG_PROVIDERS=ollama,openai,anthropic,google`. A provider-specific failure does not hide healthy providers, and the unchanged status/probe command remains the source of execution readiness.

The parser is aligned to the observed `2026.7.1-2` catalog/status structure. Live per-model readiness comes primarily from `auth.probes.results`; provider authentication and quota context remain in `auth.oauth` and `auth.providers`. Revalidate it after OpenClaw upgrades because JSON fields may evolve. The built-in approved pool is exactly Qwen 3 8B, GPT-5.6 Sol, Claude Opus 4.8, Claude Sonnet 5, and Gemini 3.1 Pro Preview. Extra catalog rows remain ineligible unless deliberately added to policy configuration and independently probe-verified. Selection-only is the normal integration from an existing OpenClaw workflow; external applications may use select-and-execute.

## Authentication ownership

OpenClaw owns all provider authentication. The verified GPT route uses OpenAI subscription OAuth. The verified Opus route and configured Sonnet route reuse the authenticated Claude CLI through OpenClaw. Gemini uses the Google credential path configured in OpenClaw. The Decision Engine reads only sanitized readiness, expiry, and quota status; it does not read or store provider credential values.

Run setup and authentication commands as the same operating-system user that runs OpenClaw. Provider configuration alone is insufficient: confirm each intended model with `openclaw models status --probe --json`. Never copy provider tokens into Decision Engine configuration.
