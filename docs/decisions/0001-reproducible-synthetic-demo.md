# ADR 0001: Use a reproducible synthetic demo dataset

- **Date:** 2026-08-30
- **Status:** Accepted

## Context

The original demo contained only 14 rows spread across six series. Fourteen-day lag and rolling features removed every row, making both local execution and CI fail.

## Decision

Bundle one year of deterministic synthetic daily data generated from a documented script and fixed seed. Validate that temporal train and test partitions are non-empty and fail with a clear message when input history is insufficient.

## Consequences

The pipeline is reproducible and usable without network access. The data is not representative of operational retail behavior, so documentation and output claims must identify it as synthetic and avoid claims about business impact.
