# Baseline Forecast Model Card

## Intended use

Demonstrate a small chronological forecasting comparison and generate example inventory-error summaries from the bundled synthetic data.

## Models

- Seven-day seasonal naive forecast
- Seven-day rolling mean
- Scikit-learn linear regression using calendar, event, price, lag, and rolling features

## Evaluation

The final 35 feature-complete dates are held out chronologically. This is a single holdout, not rolling-origin cross-validation. Current synthetic-data results are recorded in `docs/experiments/2026-08-30-baseline-repair.md`.

## Non-goals and risks

The models are not intended to automate purchasing or inventory decisions. They provide no prediction intervals, causal estimates, calibration guarantees, cold-start handling, or evidence of transfer to real stores. Stockouts affect observed sales in the generator, so sales are not equivalent to unconstrained demand.
