import pytest

from app.services.stress_simulator import (
    SCENARIOS,
    generate_stress_history,
    summarize_stress,
)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_scenario_generation(scenario):
    history = generate_stress_history(scenario)

    assert len(history) == 180
    assert history[0]["scenario"] == scenario


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_accounting_consistency(scenario):
    history = generate_stress_history(scenario)

    for row in history:
        expected = round(
            row["opening_balance_zar"] +
            row["net_cash_flow_zar"], 2
        )

        assert row["closing_balance_zar"] == expected
        assert row["supplier_arrears_zar"] >= 0


def test_deterministic_simulation():
    first = generate_stress_history(
        "declining", seed=100
    )
    second = generate_stress_history(
        "declining", seed=100
    )

    assert first == second


def test_invalid_scenario_rejected():
    with pytest.raises(ValueError):
        generate_stress_history("invalid")


def test_summary():
    history = generate_stress_history(
        "supplier_distress"
    )

    summary = summarize_stress(history)

    assert summary["observation_days"] == 180
    assert summary["final_supplier_arrears_zar"] > 0
