"""Versioned contracts. No target, identity or demographic field is a predictor."""
from __future__ import annotations

from datetime import date
from typing import Annotated, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "ieri.financial-features.v1"
TARGET_VERSION = "ieri.forward-stress.v1"
SIMULATOR_VERSION = "ieri.cash-ledger.v1"
HISTORY_DAYS, HORIZON_DAYS = 90, 30
NOTICE = (
    "Research demonstration trained on fictional businesses. Scores estimate a synthetic "
    "30-day financial-stress event, not loan default or a validated real-world probability. "
    "No lending approval or rejection is produced."
)
FEATURES = (
    "average_daily_sales_zar", "sales_trend_30d_pct", "net_cash_flow_zar",
    "cash_flow_volatility_zar", "negative_cash_flow_day_share", "cash_outflow_ratio",
    "liquidity_coverage_days", "digital_payment_share", "vas_revenue_share",
    "payment_channel_entropy", "supplier_due_payment_ratio",
)

Number = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Money = Annotated[Number, Field(ge=0, le=1e10)]
Share = Annotated[Number, Field(ge=0, le=1)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class FinancialFeatures(StrictModel):
    average_daily_sales_zar: Annotated[Money, Field(gt=0)]
    sales_trend_30d_pct: Annotated[Number, Field(ge=-100, le=1e14)]
    net_cash_flow_zar: Annotated[Number, Field(ge=-1e12, le=1e12)]
    cash_flow_volatility_zar: Money
    negative_cash_flow_day_share: Share
    cash_outflow_ratio: Annotated[Number, Field(ge=0, le=1e12)]
    liquidity_coverage_days: Annotated[Number, Field(ge=0, le=1e12)]
    digital_payment_share: Share
    vas_revenue_share: Share
    payment_channel_entropy: Share
    supplier_due_payment_ratio: Annotated[Number, Field(ge=0, le=1e12)]

    @field_validator("*", mode="before")
    @classmethod
    def reject_coercion(cls, value):
        if isinstance(value, (bool, str)):
            raise ValueError("Expected a finite JSON number, not a boolean or string")
        return value

    @model_validator(mode="after")
    def cash_identity(self):
        receipts = HISTORY_DAYS * self.average_daily_sales_zar
        expected = receipts * (1 - self.cash_outflow_ratio)
        if not np.isclose(self.net_cash_flow_zar, expected, atol=0.05, rtol=1e-6):
            raise ValueError("Net cash flow must equal 90-day receipts minus actual cash outflows")
        return self


class ScoreRequest(StrictModel):
    schema_version: Literal[SCHEMA_VERSION]
    business_id: Annotated[str, Field(min_length=1, max_length=100)]
    observation_end: date
    features: FinancialFeatures


class DailyRecord(StrictModel):
    business_id: Annotated[str, Field(min_length=1, max_length=100)]
    date: date
    opening_balance_zar: Money
    cash_sales_zar: Money
    card_sales_zar: Money
    mobile_sales_zar: Money
    gross_sales_zar: Money
    vas_sales_zar: Money
    supplier_due_zar: Money
    supplier_paid_zar: Money
    operating_due_zar: Money
    operating_expenses_zar: Money
    owner_draw_zar: Money
    net_cash_flow_zar: Annotated[Number, Field(ge=-1e12, le=1e12)]
    closing_balance_zar: Money
    supplier_arrears_zar: Money
    operating_arrears_zar: Money

    @field_validator("*", mode="before")
    @classmethod
    def reject_bool(cls, value):
        if isinstance(value, bool):
            raise ValueError("Boolean values are not financial observations")
        return value


class HistoryRequest(StrictModel):
    history: Annotated[list[DailyRecord], Field(min_length=90, max_length=90)]


def feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Reject missing/extra columns, bad numbers and inconsistent feature semantics."""
    if len(frame) == 0 or len(frame.columns) != len(FEATURES) or set(frame) != set(FEATURES):
        raise ValueError("Expected a nonempty frame with exactly the 11 versioned financial features")
    records = [FinancialFeatures.model_validate(row).model_dump() for row in frame.to_dict("records")]
    return pd.DataFrame(records, columns=FEATURES, index=frame.index, dtype=float)
