#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
readonly PROJECT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd -P)

if ! cd "$PROJECT_DIR"; then
    printf 'Error: could not enter %s.\n' "$PROJECT_DIR" >&2
    exit 1
fi

export ADE_OPENCLAW_ENABLED=true
export ADE_OPENCLAW_CATALOG_PROVIDERS=ollama,openai,anthropic,google
export ADE_OPENCLAW_CATALOG_TIMEOUT_SECONDS=60

demo_output=$(mktemp "${TMPDIR:-/tmp}/ade-live-demo.XXXXXX")
trap 'rm -f "$demo_output"' EXIT

run_demo() {
    local title=$1
    local expected_model=$2
    local prompt=$3
    shift 3

    printf '\n============================================================\n'
    printf '%s\n' "$title"
    printf '============================================================\n\n'

    : >"$demo_output"
    set +e
    uv run ai-decision-engine "$prompt" --select-only --show-rejections "$@" | tee "$demo_output"
    local cli_status=${PIPESTATUS[0]}
    set -e

    if ((cli_status != 0)); then
        printf 'Error: %s failed with exit status %d.\n' "$title" "$cli_status" >&2
        exit "$cli_status"
    fi

    if ! grep -Fqx "Selected: ${expected_model} (${expected_model%%/*})" "$demo_output"; then
        printf 'Error: %s did not select expected model %s.\n' "$title" "$expected_model" >&2
        exit 1
    fi

    printf '\nVerified expected selection: %s\n' "$expected_model"
}

clear

run_demo \
    'Demo 1 — Local routing' \
    'ollama/qwen3:8b' \
    'Reply briefly with one sentence explaining local-first model routing.'

sleep 2

run_demo \
    'Demo 2 — High reasoning' \
    'openai/gpt-5.6-sol' \
    'Evaluate three monetization strategies for an AI productivity platform. Compare risks, pricing power, and execution complexity, then recommend a staged launch strategy with explicit tradeoffs.' \
    --allow-cloud

sleep 2

run_demo \
    'Demo 3 — Extreme reasoning' \
    'anthropic/claude-opus-4-8' \
    'Review the architecture and compare three deployment approaches. Assess privacy, security, reliability, and failure-mode risks; evaluate cost, latency, scalability, and operational tradeoffs; design a phased migration with rollback criteria; then synthesize the evidence and recommend an implementation strategy.' \
    --allow-cloud

printf '\nAll three routing decisions matched the expected models.\n'
