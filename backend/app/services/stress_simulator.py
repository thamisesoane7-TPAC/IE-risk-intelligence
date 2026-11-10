"""IERI synthetic stress scenarios.

Demonstration data only. Not calibrated to real defaults.
"""

import random
from datetime import date, timedelta

SCENARIOS = (
    "stable",
    "seasonal",
    "declining",
    "liquidity_stress",
    "supplier_distress",
    "recovery",
)


def generate_stress_history(
    scenario: str,
    seed: int = 42,
    days: int = 180,
    base_sales: float = 1500.0,
) -> list[dict]:
    if scenario not in SCENARIOS:
        raise ValueError("Unknown financial scenario")

    if days < 90:
        raise ValueError("At least 90 days required")

    if base_sales <= 0:
        raise ValueError("base_sales must be positive")

    rng = random.Random(seed)
    balance = base_sales * 5
    supplier_arrears = 0.0
    records = []

    for i in range(days):
        progress = i / max(days - 1, 1)
        multiplier = 1.0

        if scenario == "seasonal":
            import math
            multiplier = 1 + .35 * math.sin(
                2 * math.pi * i / 30
            )

        elif scenario == "declining":
            multiplier = 1 - .65 * progress

        elif scenario == "liquidity_stress":
            multiplier = 1 - .40 * progress

        elif scenario == "supplier_distress":
            multiplier = 1 - .25 * progress

        elif scenario == "recovery":
            multiplier = (
                1 - 1.2 * progress
                if progress < .5
                else .4 + 1.2 * (progress - .5)
            )

        sales = round(
            max(
                0,
                rng.gauss(
                    base_sales * multiplier,
                    base_sales * .12
                )
            ), 2
        )

        supplier_due = round(sales * .55, 2)

        operating_cost = round(
            base_sales * (
                .62 if scenario == "liquidity_stress"
                else .32
            ), 2
        )

        payment_fraction = 1.0

        if scenario == "supplier_distress":
            payment_fraction = max(
                .30, 1 - .65 * progress
            )

        supplier_paid = round(
            supplier_due * payment_fraction, 2
        )

        supplier_arrears = round(
            supplier_arrears +
            supplier_due -
            supplier_paid, 2
        )

        net_cash_flow = round(
            sales - supplier_paid - operating_cost, 2
        )

        opening_balance = balance
        balance = round(
            balance + net_cash_flow, 2
        )

        records.append({
            "date": (
                date(2026, 1, 1) +
                timedelta(days=i)
            ).isoformat(),
            "scenario": scenario,
            "sales_zar": sales,
            "supplier_due_zar": supplier_due,
            "supplier_paid_zar": supplier_paid,
            "supplier_arrears_zar": supplier_arrears,
            "operating_cost_zar": operating_cost,
            "net_cash_flow_zar": net_cash_flow,
            "opening_balance_zar": opening_balance,
            "closing_balance_zar": balance,
        })

    return records


def summarize_stress(history: list[dict]) -> dict:
    if not history:
        raise ValueError("History cannot be empty")

    negative_days = sum(
        r["net_cash_flow_zar"] < 0
        for r in history
    )

    return {
        "scenario": history[0]["scenario"],
        "observation_days": len(history),
        "negative_cash_flow_days": negative_days,
        "negative_cash_flow_share": round(
            negative_days / len(history), 4
        ),
        "final_balance_zar": history[-1][
            "closing_balance_zar"
        ],
        "final_supplier_arrears_zar": history[-1][
            "supplier_arrears_zar"
        ],
        "total_sales_zar": round(
            sum(r["sales_zar"] for r in history), 2
        ),
    }


if __name__ == "__main__":
    import json

    for scenario in SCENARIOS:
        history = generate_stress_history(scenario)
        print(json.dumps(
            summarize_stress(history),
            indent=2
        ))
