"""Shared feature schema and safe preprocessing for credit scoring."""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "amount",
    "term_months",
    "revenue",
    "net_profit",
    "employees",
    "profit_margin",
    "business_age_months",
    "payment_to_monthly_revenue",
    "revenue_discrepancy",
    "debt_load",
    "overdue_days",
    "credit_history_months",
    "tax_debt",
]
CATEGORICAL_FEATURES = ["product", "legal_form", "industry"]
BOOLEAN_FEATURES = ["collateral", "mass_address", "unreliable", "active"]
MODEL_CATEGORICAL_FEATURES = CATEGORICAL_FEATURES + BOOLEAN_FEATURES
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES

# Categorical nulls must be represented as strings for CatBoost.
MISSING_CATEGORY = "__MISSING__"


def _to_number(value: Any):
    """Preserve unknown values as NaN; never silently turn null into zero."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    try:
        result = float(value)
        return result if np.isfinite(result) else np.nan
    except (TypeError, ValueError):
        return np.nan


def prepare_features(data: pd.DataFrame | dict) -> pd.DataFrame:
    """Return a model-ready frame in a stable column order.

    Supports a single input dict or a DataFrame. Missing columns are filled
    with missing values, not zeros. Numeric missingness is preserved as NaN.
    """
    if isinstance(data, dict):
        frame = pd.DataFrame([data])
    else:
        frame = data.copy()

    for col in FEATURES:
        if col not in frame.columns:
            frame[col] = None

    frame = frame[FEATURES].copy()

    for col in NUMERIC_FEATURES:
        frame[col] = frame[col].map(_to_number).astype(float)

    for col in BOOLEAN_FEATURES:
        # Preserve unknown booleans as a distinct category.
        def bool_to_cat(value):
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return MISSING_CATEGORY
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"true", "1", "yes", "да"}:
                    return "true"
                if lowered in {"false", "0", "no", "нет"}:
                    return "false"
                return MISSING_CATEGORY
            if isinstance(value, (bool, np.bool_)):
                return "true" if value else "false"
            if isinstance(value, (int, np.integer)) and value in (0, 1):
                return "true" if value else "false"
            return MISSING_CATEGORY

        frame[col] = frame[col].map(bool_to_cat).astype(str)

    for col in CATEGORICAL_FEATURES:
        frame[col] = frame[col].map(
            lambda x: MISSING_CATEGORY
            if x is None or (isinstance(x, float) and np.isnan(x)) or str(x).strip() == ""
            else str(x)
        ).astype(str)

    return frame


def validate_application(data: dict) -> list[str]:
    """Lightweight input checks; unknown optional fields remain permitted."""
    errors = []
    for field in ("amount", "term_months", "revenue", "net_profit", "employees"):
        value = data.get(field)
        if value is not None:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                errors.append(f"{field} must be numeric or null")
                continue
            if not np.isfinite(numeric):
                errors.append(f"{field} must be finite or null")
            elif field in ("amount", "term_months", "revenue", "employees") and numeric < 0:
                errors.append(f"{field} must be >= 0")
    return errors
