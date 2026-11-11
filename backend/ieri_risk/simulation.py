"""Fictional cash ledgers with persistent behaviour and unobserved future shocks."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from .contracts import DailyRecord, HISTORY_DAYS, HORIZON_DAYS
from .features import validate_history

SCENARIOS = ("ordinary", "demand_drop", "supplier_inflation", "operational_outage", "collection_delay", "compound")
SECTORS = ("spaza", "food_vendor", "salon", "small_retailer", "other_microenterprise")


def simulate_business(index: int, seed: int = 20261110, *, challenge: bool = False,
                      future_seed: int | None = None) -> tuple[pd.DataFrame, dict]:
    if index < 0 or seed < 0 or (future_seed is not None and future_seed < 0):
        raise ValueError("Seeds and business indices must be nonnegative")
    latent = np.random.default_rng(np.random.SeedSequence([seed, index, 0]))
    history_rng = np.random.default_rng(np.random.SeedSequence([seed, index, 1]))
    future_rng = np.random.default_rng(np.random.SeedSequence([seed if future_seed is None else future_seed, index, 2]))
    business_id = f"{'challenge' if challenge else 'synthetic'}-{seed}-{index:06d}"
    sector = str(latent.choice(SECTORS))  # audit only: not a predictor
    base = float(np.clip(latent.lognormal(np.log(1400), 0.65), 200, 9000))
    supplier_ratio = float(latent.uniform(0.38, 0.78))
    operating_ratio = float(latent.uniform(0.08, 0.23))
    draw_ratio = float(latent.uniform(0.04, 0.19))
    trend = float(latent.normal(0, 0.0016))
    noise = float(latent.uniform(0.08, 0.38))
    weekend_effect = float(latent.uniform(-0.22, 0.48))
    channels = latent.dirichlet([2.5, 1.2, 1.5])
    vas_share = float(latent.uniform(0, 0.12))
    reserve_days = float(latent.uniform(2.0, 17.0))
    cash = round(base * float(latent.uniform(2, 18)), 2)
    supplier_arrears = operating_arrears = 0.0
    receivables: dict[int, float] = {}
    state = 0.0
    # History can decline temporarily and recover, or decline persistently.
    dip_start = int(latent.integers(35, 70))
    dip_duration = int(latent.integers(8, 45))
    dip_severity = float(latent.uniform(0, 0.50))
    future_scenario = str(future_rng.choice(SCENARIOS, p=[.25, .20, .15, .12, .13, .15]))
    shock_start = int(future_rng.integers(1, 13))
    severity = float(future_rng.uniform(0.10, 0.65)) * (1.45 if challenge else 1.0)
    shock_duration = int(future_rng.integers(7, 27))
    future_trend = float(future_rng.normal(-.002 if challenge else 0, .003))
    # Unobserved recoveries and tail expenses prevent an almost deterministic
    # continuation of today's liquidity state. These are explicit assumptions,
    # not empirically estimated event frequencies.
    recovery_multiplier = float(future_rng.uniform(1.25, 1.85)) if future_rng.random() < .20 else 1.0
    tail_day = int(future_rng.integers(3, 21)) if future_rng.random() < (.15 if challenge else .08) else -1
    tail_expense = round(base * float(future_rng.uniform(10, 22)), 2) if tail_day >= 0 else 0.0
    records = []
    start = date(2026, 1, 1)
    for day in range(HISTORY_DAYS + HORIZON_DAYS):
        rng = history_rng if day < HISTORY_DAYS else future_rng
        ahead = day - HISTORY_DAYS
        shocked = ahead >= shock_start and ahead < shock_start + shock_duration
        state = .62 * state + float(rng.normal(0, noise))
        weekday = (start + timedelta(days=day)).weekday()
        demand = base * np.exp(np.clip(trend * day + state, -2.0, 1.5))
        demand *= 1 + weekend_effect * (weekday >= 5)
        if dip_start <= day < dip_start + dip_duration:
            demand *= 1 - dip_severity
        if ahead >= 0:
            demand *= np.exp(future_trend * ahead)
            demand *= 1 + (recovery_multiplier - 1) * min(1.0, (ahead + 1) / 8)
        cost_multiplier = 1.0
        if shocked:
            if future_scenario in ("demand_drop", "compound"):
                demand *= max(.04, 1 - severity)
            if future_scenario in ("supplier_inflation", "compound"):
                cost_multiplier += severity * .85
            if future_scenario == "operational_outage":
                demand *= .07 if ahead < shock_start + max(2, int(severity * 14)) else .9
        # Unscheduled daily closures occur in both periods; the future is not a new regime by definition.
        if rng.random() < .025:
            demand *= .04
        sales = round(max(demand, 0), 2)
        if shocked and future_scenario in ("collection_delay", "compound"):
            delayed = round(sales * min(.8, severity), 2)
            sales -= delayed
            release = day + int(rng.integers(5, 16))
            receivables[release] = receivables.get(release, 0) + delayed
        sales = round(sales + receivables.pop(day, 0), 2)
        operating_due = round(base * operating_ratio * (.7 + (weekday == 0) * 2.1), 2)
        supplier_due = round(base * supplier_ratio * 7 * cost_multiplier, 2) if weekday == 2 else 0.0
        # Repairs are unscheduled obligations, not generated directly from the label.
        if rng.random() < (.014 if challenge and ahead >= 0 else .007):
            operating_due += round(base * float(rng.uniform(.5, 4)), 2)
        if ahead >= 0 and ahead == tail_day:
            operating_due += tail_expense
        opening = cash
        available = round(cash + sales, 2)
        operating_arrears = round(operating_arrears + operating_due, 2)
        supplier_arrears = round(supplier_arrears + supplier_due, 2)
        operating_paid = min(available, operating_arrears)
        operating_arrears = round(operating_arrears - operating_paid, 2)
        available = round(available - operating_paid, 2)
        supplier_paid = min(available, supplier_arrears)
        supplier_arrears = round(supplier_arrears - supplier_paid, 2)
        available = round(available - supplier_paid, 2)
        draw = min(available, round(sales * draw_ratio, 2))
        if weekday == 6:
            draw = round(draw + max(0, available - draw - base * reserve_days) * .6, 2)
        cash = round(available - draw, 2)
        card = round(sales * channels[1], 2)
        mobile = round(sales * channels[2], 2)
        cash_sales = round(sales - card - mobile, 2)
        records.append({
            "business_id": business_id, "date": (start + timedelta(days=day)).isoformat(),
            "opening_balance_zar": opening, "cash_sales_zar": cash_sales,
            "card_sales_zar": card, "mobile_sales_zar": mobile, "gross_sales_zar": sales,
            "vas_sales_zar": round(sales * vas_share, 2), "supplier_due_zar": supplier_due,
            "supplier_paid_zar": supplier_paid, "operating_due_zar": operating_due,
            "operating_expenses_zar": operating_paid, "owner_draw_zar": draw,
            "net_cash_flow_zar": round(cash - opening, 2), "closing_balance_zar": cash,
            "supplier_arrears_zar": supplier_arrears, "operating_arrears_zar": operating_arrears,
        })
    return pd.DataFrame(records), {
        "business_id": business_id, "sector": sector,
        "future_scenario": future_scenario, "future_severity": severity,
        "future_recovery_multiplier": recovery_multiplier,
        "future_tail_expense_zar": tail_expense,
        "population": "shifted_challenge" if challenge else "development",
    }


def forward_outcome(history: pd.DataFrame, future: pd.DataFrame) -> dict:
    """Future shortfalls, with FIFO settlement and the opening arrears carried forward."""
    h = validate_history(history)
    if len(future) != HORIZON_DAYS:
        raise ValueError("Exactly 30 future outcome days are required")
    f = pd.DataFrame([DailyRecord.model_validate(r).model_dump() for r in future.to_dict("records")])
    f["date"] = pd.to_datetime(f["date"])
    f = f.sort_values("date").reset_index(drop=True)
    if f.business_id.nunique() != 1 or f.business_id.iloc[0] != h.business_id.iloc[0]:
        raise ValueError("Outcome business does not match the historical business")
    expected = pd.date_range(h.date.iloc[-1] + pd.Timedelta(days=1), periods=HORIZON_DAYS)
    if not f.date.equals(pd.Series(expected, name="date")):
        raise ValueError("The outcome window must immediately follow, and never overlap, history")
    # Reuse complete accounting validation on a rolling 90-day block crossing the boundary.
    validate_history(pd.concat([h.iloc[-60:], f], ignore_index=True))
    supplier_old = float(h.supplier_arrears_zar.iloc[-1])
    operating_old = float(h.operating_arrears_zar.iloc[-1])
    opening_arrears = supplier_old + operating_old
    missed_days = 0
    for row in f.itertuples(index=False):
        supplier_today_paid = max(0, row.supplier_paid_zar - supplier_old)
        operating_today_paid = max(0, row.operating_expenses_zar - operating_old)
        due_today = row.supplier_due_zar + row.operating_due_zar
        shortfall = max(0, row.supplier_due_zar - supplier_today_paid) + max(0, row.operating_due_zar - operating_today_paid)
        missed_days += int(due_today > 0 and shortfall > max(1.0, .05 * due_today))
        supplier_old, operating_old = row.supplier_arrears_zar, row.operating_arrears_zar
    peak_new_arrears = max(0.0, float((f.supplier_arrears_zar + f.operating_arrears_zar).max()) - opening_arrears)
    arrears_limit = 7 * max(1.0, float((h.supplier_due_zar + h.operating_due_zar).mean()))
    return {
        "business_id": h.business_id.iloc[0],
        "outcome_start": f.date.iloc[0].date().isoformat(),
        "outcome_end": f.date.iloc[-1].date().isoformat(),
        "future_stress_30d": int(missed_days >= 5 or peak_new_arrears > arrears_limit),
        "future_missed_payment_days": missed_days,
        "future_peak_incremental_arrears_zar": peak_new_arrears,
        "arrears_threshold_zar": arrears_limit,
    }
