# Baseline pipeline verification — 2026-08-30

## Motivation

The original 14-row demo dataset left no rows after 14-day lag construction, so the checked-in workflow failed before training. The demo data was replaced with a deterministic 2,190-row synthetic dataset and the temporal split was made adaptive and explicitly validated.

## Configuration

- Python used locally: 3.14.7
- CI target: Python 3.11
- Dataset seed: `20260409`
- Split: final 35 feature-complete dates as test data
- Features: calendar variables, promotion/holiday/stockout indicators, price, lags 1/7/14, rolling means 7/14
- Models: naive 7-day lag, 7-day rolling mean, linear regression

## Results

| Model | MAE | RMSE | MAPE |
| --- | ---: | ---: | ---: |
| LinearRegression | 2.47 | 3.10 | 6.43% |
| RollingMean_7D | 3.16 | 4.32 | 8.63% |
| Naive_7D | 3.95 | 5.66 | 10.65% |

Verification commands:

```bash
python scripts/generate_demo_data.py
python -m pytest -q
python -m src.forecasting
```

Observed result: `3 passed`; the pipeline completed and regenerated all metric and figure artifacts.

## Conclusion

The repository now demonstrates a working end-to-end pipeline. The numbers are evidence for code-path correctness on synthetic data only and should not be generalized to real inventory decisions.
