"""Generate the deterministic synthetic retail dataset bundled with the project."""

from __future__ import annotations

import csv
import math
import random
from datetime import date, timedelta
from pathlib import Path

SEED = 20260409
DAYS = 365
OUTPUT = Path(__file__).resolve().parents[1] / "data" / "demo" / "retail_demand_demo.csv"

BASE_DEMAND = {
    ("S1", "P1"): 38,
    ("S1", "P2"): 33,
    ("S1", "P3"): 28,
    ("S2", "P1"): 44,
    ("S2", "P2"): 37,
    ("S2", "P3"): 27,
}
PRICES = {"P1": 9.99, "P2": 12.50, "P3": 7.75}


def generate_rows() -> list[dict[str, object]]:
    rng = random.Random(SEED)
    start = date(2024, 1, 1)
    rows: list[dict[str, object]] = []

    for day_index in range(DAYS):
        current = start + timedelta(days=day_index)
        weekend = int(current.weekday() >= 5)
        holiday = int((current.month, current.day) in {(1, 1), (7, 4), (11, 28), (12, 25)})
        annual = 4.5 * math.sin(2 * math.pi * day_index / DAYS)

        for series_index, ((store_id, product_id), base) in enumerate(BASE_DEMAND.items()):
            promo = int((day_index + 11 * series_index) % 37 in {0, 1, 2})
            stockout = int((day_index + 17 * series_index) % 113 == 0)
            weekly = 3.0 if current.weekday() in {4, 5} else -1.0 if current.weekday() == 0 else 0.0
            demand = base + 0.012 * day_index + annual + weekly + 7 * promo + 9 * holiday
            demand += rng.gauss(0, 2.8)
            if stockout:
                demand *= 0.45

            sales = max(0, round(demand))
            unit_price = round(PRICES[product_id] * (0.90 if promo else 1.0), 2)
            rows.append(
                {
                    "date": current.isoformat(),
                    "store_id": store_id,
                    "product_id": product_id,
                    "sales": sales,
                    "promo": promo,
                    "holiday": holiday,
                    "weekend": weekend,
                    "stockout": stockout,
                    "unit_price": f"{unit_price:.2f}",
                    "revenue": f"{sales * unit_price:.2f}",
                }
            )
    return rows


def main() -> None:
    rows = generate_rows()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} deterministic rows to {OUTPUT}")


if __name__ == "__main__":
    main()
