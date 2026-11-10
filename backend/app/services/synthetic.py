"""Reproducible fictional financial data for IERI."""

import csv
import random
from datetime import date, timedelta
from pathlib import Path

PROFILES = (
    {"id": "IERI-001", "sector": "spaza_shop", "sales": 2200, "supplier": .55, "operating": .20, "digital": .16, "vas": .14},
    {"id": "IERI-002", "sector": "food_vendor", "sales": 1350, "supplier": .43, "operating": .28, "digital": .25, "vas": .02},
    {"id": "IERI-003", "sector": "salon", "sales": 950, "supplier": .18, "operating": .50, "digital": .57, "vas": .00},
    {"id": "IERI-004", "sector": "repair_shop", "sales": 1700, "supplier": .38, "operating": .44, "digital": .42, "vas": .02},
    {"id": "IERI-005", "sector": "street_vendor", "sales": 620, "supplier": .66, "operating": .31, "digital": .06, "vas": .05},
)


def generate_ledger(
    seed: int = 42,
    days: int = 120,
    start: date = date(2026, 1, 1)
) -> list[dict]:
    if days < 60:
        raise ValueError("at least 60 days are required")

    rng = random.Random(seed)
    rows = []

    for profile in PROFILES:
        balance = round(profile["sales"] * 2.5, 2)

        for offset in range(days):
            day = start + timedelta(days=offset)
            weekend = 1.12 if day.weekday() >= 5 else 1.0

            disruption = (
                rng.uniform(.35, .75)
                if rng.random() < .04 else 1.0
            )

            sales = round(max(0, rng.gauss(
                profile["sales"] * weekend * disruption,
                profile["sales"] * .17
            )), 2)

            digital = min(
                .90,
                max(.01, rng.gauss(profile["digital"], .045))
            )

            card = round(
                sales * digital * rng.uniform(.25, .70), 2
            )
            mobile = round(sales * digital, 2) - card
            cash = round(sales - card - mobile, 2)

            # VAS is a subset of gross sales, not extra income.
            vas = round(min(
                sales,
                sales * profile["vas"] * rng.uniform(.70, 1.30)
            ), 2)

            supplier_due = round(
                sales * profile["supplier"] *
                rng.uniform(.86, 1.14), 2
            )

            supplier_paid = round(
                supplier_due * (
                    rng.uniform(.45, .85)
                    if rng.random() < .10 else 1.0
                ), 2
            )

            opex = round(
                sales * profile["operating"] *
                rng.uniform(.84, 1.18), 2
            )

            net = round(sales - supplier_paid - opex, 2)
            opening = balance
            balance = round(opening + net, 2)

            rows.append({
                "business_id": profile["id"],
                "sector": profile["sector"],
                "date": day.isoformat(),
                "opening_balance_zar": opening,
                "cash_sales_zar": cash,
                "card_sales_zar": card,
                "mobile_sales_zar": mobile,
                "gross_sales_zar": sales,
                "vas_sales_zar": vas,
                "supplier_due_zar": supplier_due,
                "supplier_paid_zar": supplier_paid,
                "operating_expenses_zar": opex,
                "net_cash_flow_zar": net,
                "closing_balance_zar": balance
            })

    return rows


def save_csv(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    output = (
        Path(__file__).resolve().parents[2]
        / "data" / "synthetic_daily_ledger.csv"
    )
    ledger = generate_ledger()
    save_csv(ledger, output)
    print(f"Created {len(ledger)} fictional records")
    print(f"Saved to {output}")
