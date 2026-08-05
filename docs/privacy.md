# Privacy policy

Privacy is fail-safe by default:

- `local_only` always excludes cloud models.
- `cloud_allowed` or `public` must be explicitly supplied before cloud can be considered.
- Omitting classification defaults to `local_only`.
- Sensitive-content heuristics may increase a request to local-only, but never grant cloud access.

Explicit cloud permission is included in response metadata and routing logs. Heuristics are defense in depth, not data-loss prevention. Callers should classify data using organizational policy before sending a request and should never interpret this router as a complete privacy scanner.

ChatGPT/Codex OAuth, Claude subscription reuse, and Gemini authentication still execute in provider clouds. OpenClaw authentication does not make those routes local. The Gateway operator token is never persisted or logged and must stay on a private/protected network path.
