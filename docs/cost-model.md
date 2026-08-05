# Cost model

Local Ollama inference has zero API spend. Electricity, hardware depreciation, and operations are separate costs.

Billing modes are local, subscription, free-tier, and API. Subscription and free-tier routes have zero incremental API estimate only while their externally managed quota is usable; they are not unlimited or genuinely free. API routes require explicit per-million input/output pricing and conservative token estimates.

Before an API dispatch, the engine uses `BEGIN IMMEDIATE` to reserve the maximum estimated cost in SQLite. The `$10` default cap applies only to metered API spend, not included subscription allowance.

Success reconciles to provider-reported cost when available, otherwise the final estimate. A failed call with a known cost records that cost. A possibly charged failure conservatively finalizes the full reservation. A reservation is released only when the provider contract confirms no charge. Cumulative finalized costs constrain later fallbacks under the same request budget.

Database corruption, missing schema, locking failure, or I/O errors raise `LedgerUnavailableError`; cloud execution does not proceed when budget state cannot be trusted. Active reservations intentionally continue reducing available budget after an unexpected process termination. Operational recovery tooling for stale reservations remains future work.
