import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from ieri_risk.contracts import FEATURES, FinancialFeatures, feature_frame
from ieri_risk.features import calculate_features, validate_history
from ieri_risk.simulation import forward_outcome, simulate_business


def sample():
    return simulate_business(3, seed=33)[0]


def test_generator_reproducible_and_rng_independent():
    a, _ = simulate_business(12, 7)
    b, _ = simulate_business(12, 7)
    pd.testing.assert_frame_equal(a, b)
    np.random.seed(999)
    c, _ = simulate_business(12, 7, future_seed=987)
    pd.testing.assert_frame_equal(a.iloc[:90], c.iloc[:90])
    assert calculate_features(a.iloc[:90]) == calculate_features(c.iloc[:90])
    assert not a.iloc[90:].equals(c.iloc[90:])


def test_exact_feature_allowlist_no_identity_or_future():
    features = calculate_features(sample().iloc[:90])
    assert tuple(features) == FEATURES
    assert not set(features) & {"business_id", "sector", "age", "gender", "race", "future_scenario", "future_stress_30d"}


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True, "1200", None])
def test_bad_numeric_features_rejected(value):
    features = calculate_features(sample().iloc[:90])
    features["average_daily_sales_zar"] = value
    with pytest.raises(ValidationError):
        FinancialFeatures.model_validate(features)


@pytest.mark.parametrize("field", ["race", "business_id", "future_stress_30d"])
def test_extra_predictors_rejected(field):
    features = calculate_features(sample().iloc[:90])
    features[field] = 0.0
    with pytest.raises(ValueError):
        feature_frame(pd.DataFrame([features]))


def test_missing_feature_and_cash_inconsistency_rejected():
    features = calculate_features(sample().iloc[:90])
    bad = dict(features)
    bad.pop("digital_payment_share")
    with pytest.raises(ValueError):
        feature_frame(pd.DataFrame([bad]))
    features["net_cash_flow_zar"] += 500
    with pytest.raises(ValidationError):
        FinancialFeatures.model_validate(features)


@pytest.mark.parametrize("mutation", ["missing_day", "duplicate_day", "mixed_business", "cash_identity", "arrears_identity", "zero_receipts"])
def test_invalid_ledger_rejected(mutation):
    history = sample().iloc[:90].copy()
    if mutation == "missing_day":
        history = history.iloc[:-1]
    elif mutation == "duplicate_day":
        history.loc[1, "date"] = history.loc[0, "date"]
    elif mutation == "mixed_business":
        history.loc[4, "business_id"] = "other"
    elif mutation == "cash_identity":
        history.loc[4, "closing_balance_zar"] += 500
    elif mutation == "arrears_identity":
        history.loc[4, "supplier_arrears_zar"] += 500
    else:
        history["gross_sales_zar"] = 0.0
    with pytest.raises(ValueError):
        validate_history(history)


def test_outcome_cannot_overlap_history():
    data = sample()
    with pytest.raises(ValueError, match="immediately follow"):
        forward_outcome(data.iloc[:90], data.iloc[89:119])


def test_refuse_future_rows_as_history():
    with pytest.raises(ValueError, match="Exactly 90"):
        calculate_features(sample())


def controlled_ledger(missed_days: int):
    rows, cash, supplier_arrears, operating_arrears = [], 1000.0, 0.0, 0.0
    for day in range(120):
        missed = 90 <= day < 90 + missed_days
        supplier_paid, operating_paid = (0.0, 0.0) if missed else (30.0, 20.0)
        supplier_arrears += 30 - supplier_paid
        operating_arrears += 20 - operating_paid
        opening = cash
        cash += 100 - supplier_paid - operating_paid
        rows.append({"business_id": "control", "date": (pd.Timestamp("2026-01-01") + pd.Timedelta(days=day)).date(),
                     "opening_balance_zar": opening, "cash_sales_zar": 100.0, "card_sales_zar": 0.0,
                     "mobile_sales_zar": 0.0, "gross_sales_zar": 100.0, "vas_sales_zar": 0.0,
                     "supplier_due_zar": 30.0, "supplier_paid_zar": supplier_paid,
                     "operating_due_zar": 20.0, "operating_expenses_zar": operating_paid,
                     "owner_draw_zar": 0.0, "net_cash_flow_zar": cash - opening,
                     "closing_balance_zar": cash, "supplier_arrears_zar": supplier_arrears,
                     "operating_arrears_zar": operating_arrears})
    return pd.DataFrame(rows)


def test_target_is_future_payment_event_not_feature_threshold():
    healthy, stressed = controlled_ledger(0), controlled_ledger(5)
    assert calculate_features(healthy.iloc[:90]) == calculate_features(stressed.iloc[:90])
    assert forward_outcome(healthy.iloc[:90], healthy.iloc[90:])["future_stress_30d"] == 0
    result = forward_outcome(stressed.iloc[:90], stressed.iloc[90:])
    assert result["future_stress_30d"] == 1
    assert result["future_missed_payment_days"] >= 5
