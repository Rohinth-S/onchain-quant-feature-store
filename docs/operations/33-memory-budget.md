# Onchain feature store operations: memory budget

## Purpose

This runbook records one production or research invariant for the onchain feature-store pipeline. It keeps event normalization, leakage-safe feature construction, walk-forward evaluation, and live RPC boundaries reproducible.

## Checks

- Validate the input event schema before feature construction and preserve timestamps, pool identifiers, and side information.
- Keep deterministic fixtures explicit; live JSON-RPC access must be selected through the documented network and credential variables.
- Record the report schema, model assumptions, and data provenance with every generated artifact.
- Treat chain reorgs, missing events, decimals, gas, and slippage as explicit uncertainty rather than silently imputing them.

## Failure handling

Stop the run when validation or provenance checks fail. Preserve the input hash, network, block range, and command line so the result can be replayed after correcting the data source.

## Verification

Run python -m pytest -q and execute the CLI on a small CSV fixture. A live run must also record the RPC endpoint class, chain identifier, block range, and adapter response status.
