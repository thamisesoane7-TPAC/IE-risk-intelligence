"""Transparent financial features; not credit scores."""

import math
from statistics import fmean, pstdev

from app.services.synthetic import generate_ledger


def calculate_features(
    ledger: list[dict],
    business_id: str
) -> dict:

    records = sorted(
        (
            row for row in ledger
            if row["business_id"] == business_id
        ),
        key=lambda row: row["date"]
    )

    if len(records) < 60:
        raise ValueError(
            "Business must have at least 60 daily records"
        )

    dates = [row["date"] for row in records]
    if len(set(dates)) != len(dates):
        raise ValueError("Duplicate dates are not allowed")

    sales = sum(
        row["gross_sales_zar"] for row in records
    )

    costs = sum(
        row["supplier_paid_zar"] +
        row["operating_expenses_zar"]
        for row in records
    )

    due = sum(
        row["supplier_due_zar"] for row in records
    )

    paid = sum(
        row["supplier_paid_zar"] for row in records
    )

    net = [
        row["net_cash_flow_zar"] for row in records
    ]

    first_30 = fmean(
        row["gross_sales_zar"]
        for row in records[:30]
    )

    last_30 = fmean(
        row["gross_sales_zar"]
        for row in records[-30:]
    )

    channels = [
        sum(row[field] for row in records)
        for field in (
            "cash_sales_zar",
            "card_sales_zar",
            "mobile_sales_zar"
        )
    ]

    fractions = (
        [amount / sales for amount in channels]
        if sales else []
    )

    entropy = (
        -sum(
            p * math.log(p)
            for p in fractions if p > 0
        ) / math.log(3)
        if fractions else 0.0
    )

    daily_cost = costs / len(records)

    return {
        "business_id": business_id,
        "observation_days": len(records),
        "average_daily_sales_zar": round(
            sales / len(records), 2
        ),
        "sales_trend_30d_pct": round(
            100 * (last_30 - first_30) / first_30, 2
        ) if first_30 else 0.0,
        "net_cash_flow_zar": round(sum(net), 2),
        "cash_flow_volatility_zar": round(
            pstdev(net), 2
        ),
        "negative_cash_flow_day_share": round(
            sum(x < 0 for x in net) / len(net), 4
        ),
        "cash_outflow_ratio": round(
            costs / sales, 4
        ) if sales else 0.0,
        "liquidity_coverage_days": round(
            max(records[-1]["closing_balance_zar"], 0)
            / daily_cost, 2
        ) if daily_cost else 0.0,
        "digital_payment_share": round(
            (channels[1] + channels[2]) / sales, 4
        ) if sales else 0.0,
        "vas_revenue_share": round(
            sum(r["vas_sales_zar"] for r in records)
            / sales, 4
        ) if sales else 0.0,
        "payment_channel_entropy": round(entropy, 4),
        "supplier_due_payment_ratio": round(
            min(paid / due, 1), 4
        ) if due else 0.0
    }


if __name__ == "__main__":
    import json

    ledger = generate_ledger()
    result = calculate_features(ledger, "IERI-001")

    print(json.dumps(result, indent=2))
