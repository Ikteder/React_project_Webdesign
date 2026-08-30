# Synthetic Retail Demand Dataset Card

- **Generated:** 2026-08-30
- **Generator:** `scripts/generate_demo_data.py`
- **Random seed:** `20260409`
- **Rows:** 2,190
- **Coverage:** 2024-01-01 through 2024-12-30
- **Series:** 2 stores × 3 products
- **License:** MIT as part of this repository; no external source data is included

## Purpose

This dataset exists to exercise the forecasting, evaluation, reporting, and dashboard paths without requiring an external download. It is not a sample of real customers, stores, products, or transactions.

## Schema

| Field | Meaning |
| --- | --- |
| `date` | Daily observation date |
| `store_id`, `product_id` | Synthetic series identifiers |
| `sales` | Generated non-negative unit sales |
| `promo`, `holiday`, `weekend`, `stockout` | Generated binary conditions |
| `unit_price` | Synthetic product price with a deterministic promotion discount |
| `revenue` | `sales × unit_price` |

## Generation and quality concerns

Demand combines a fixed series baseline, a small trend, annual and weekly effects, promotions, holidays, seeded Gaussian noise, and occasional stockout suppression. The construction is deliberately learnable and therefore likely makes model performance look better than performance on operational data. It does not model product launches, missing records, substitutions, price elasticity, store closures, supply constraints, or changing customer behavior.

The generator is the source of truth. Regenerate the CSV with:

```bash
python scripts/generate_demo_data.py
```
