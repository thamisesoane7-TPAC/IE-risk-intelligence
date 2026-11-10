import pytest

from app.services.synthetic import (
    PROFILES,
    generate_ledger
)
from app.services.features import calculate_features


def test_reproducible_ledger():
    assert generate_ledger(seed=7) == generate_ledger(seed=7)
    assert generate_ledger(seed=7) != generate_ledger(seed=8)


def test_record_count():
    records = generate_ledger(days=90)
    assert len(records) == len(PROFILES) * 90


def test_daily_accounting_invariants():
    for r in generate_ledger(days=60):
        revenue = round(
            r["cash_sales_zar"] +
            r["card_sales_zar"] +
            r["mobile_sales_zar"], 2
        )

        closing = round(
            r["opening_balance_zar"] +
            r["gross_sales_zar"] -
            r["supplier_paid_zar"] -
            r["operating_expenses_zar"], 2
        )

        assert revenue == r["gross_sales_zar"]
        assert closing == r["closing_balance_zar"]

        assert 0 <= r["vas_sales_zar"] <= revenue
        assert (
            0 <= r["supplier_paid_zar"]
            <= r["supplier_due_zar"]
        )


def test_feature_bounds():
    features = calculate_features(
        generate_ledger(), "IERI-001"
    )

    assert features["observation_days"] == 120

    for key in (
        "digital_payment_share",
        "vas_revenue_share",
        "payment_channel_entropy",
        "negative_cash_flow_day_share",
        "supplier_due_payment_ratio"
    ):
        assert 0 <= features[key] <= 1

    assert features["average_daily_sales_zar"] > 0


def test_insufficient_history_rejected():
    with pytest.raises(ValueError, match="at least 60"):
        generate_ledger(days=10)

    with pytest.raises(ValueError, match="at least 60"):
        calculate_features(
            generate_ledger(days=60), "unknown"
        )
