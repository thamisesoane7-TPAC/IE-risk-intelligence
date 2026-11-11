"""Cash-basis features computed exclusively from the 90 observed days."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import DailyRecord, FEATURES, FinancialFeatures, HISTORY_DAYS


def validate_history(history: pd.DataFrame) -> pd.DataFrame:
    if len(history) != HISTORY_DAYS:
        raise ValueError("Exactly 90 consecutive historical daily records are required")
    rows = [DailyRecord.model_validate(x).model_dump() for x in history.to_dict("records")]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    if df.business_id.nunique() != 1:
        raise ValueError("A history must contain exactly one business")
    df = df.sort_values("date").reset_index(drop=True)
    if not df.date.diff().iloc[1:].eq(pd.Timedelta(days=1)).all():
        raise ValueError("Duplicate or missing historical dates")
    receipts = df[["cash_sales_zar", "card_sales_zar", "mobile_sales_zar"]].sum(axis=1)
    outflows = df.supplier_paid_zar + df.operating_expenses_zar + df.owner_draw_zar
    checks = [
        np.allclose(receipts, df.gross_sales_zar, atol=0.05, rtol=1e-7),
        np.allclose(df.gross_sales_zar - outflows, df.net_cash_flow_zar, atol=0.05, rtol=1e-7),
        np.allclose(df.opening_balance_zar + df.net_cash_flow_zar, df.closing_balance_zar, atol=0.05, rtol=1e-7),
        np.allclose(df.opening_balance_zar.iloc[1:], df.closing_balance_zar.iloc[:-1], atol=0.05, rtol=1e-7),
        bool((df.vas_sales_zar <= df.gross_sales_zar + 0.01).all()),
    ]
    for due, paid, arrears in [
        ("supplier_due_zar", "supplier_paid_zar", "supplier_arrears_zar"),
        ("operating_due_zar", "operating_expenses_zar", "operating_arrears_zar"),
    ]:
        initial = df[arrears].iloc[0] - df[due].iloc[0] + df[paid].iloc[0]
        checks.append(initial >= -0.05)
        checks.append(np.allclose(
            df[arrears].iloc[1:].to_numpy(),
            (df[arrears].shift(1) + df[due] - df[paid]).iloc[1:].to_numpy(),
            atol=0.05, rtol=1e-7,
        ))
    if not all(checks):
        raise ValueError("Ledger does not reconcile: receipts, cash balance, payments or arrears")
    if df.gross_sales_zar.sum() <= 0:
        raise ValueError("A zero-receipts history is outside this model's feature domain")
    return df


def calculate_features(history: pd.DataFrame) -> dict[str, float]:
    df = validate_history(history)
    gross = float(df.gross_sales_zar.sum())
    prev = float(df.gross_sales_zar.iloc[-60:-30].sum())
    recent = float(df.gross_sales_zar.iloc[-30:].sum())
    channels = df[["cash_sales_zar", "card_sales_zar", "mobile_sales_zar"]].sum().to_numpy() / gross
    positive = channels[channels > 0]
    due = float((df.supplier_due_zar + df.operating_due_zar).mean())
    result = dict(zip(FEATURES, [
        gross / HISTORY_DAYS,
        100 * (recent - prev) / max(prev, 1.0),
        float(df.net_cash_flow_zar.sum()),
        float(df.net_cash_flow_zar.std(ddof=0)),
        float((df.net_cash_flow_zar < -0.005).mean()),
        float((df.supplier_paid_zar + df.operating_expenses_zar + df.owner_draw_zar).sum()) / gross,
        float(df.closing_balance_zar.iloc[-1]) / max(due, 1.0),
        float(channels[1] + channels[2]),
        float(df.vas_sales_zar.sum()) / gross,
        float(-np.sum(positive * np.log(positive)) / np.log(3)),
        float(df.supplier_paid_zar.sum()) / max(float(df.supplier_due_zar.sum()), 1.0),
    ], strict=True))
    return FinancialFeatures.model_validate(result).model_dump()
