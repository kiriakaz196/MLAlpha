"""Inference helper intended to be called by a FastAPI endpoint."""
from __future__ import annotations

from pathlib import Path
from catboost import CatBoostClassifier

try:
    from .features import prepare_features, validate_application
except ImportError:  # running this file directly from the ml directory
    from features import prepare_features, validate_application

DEFAULT_MODEL_PATH = Path(__file__).parent / "artifacts" / "credit_default_model.cbm"


class CreditScorer:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH):
        self.model = CatBoostClassifier()
        self.model.load_model(str(model_path))

    def score(self, application: dict, review_threshold: float = 0.10) -> dict:
        errors = validate_application(application)
        if errors:
            raise ValueError("; ".join(errors))

        X = prepare_features(application)
        probability = float(self.model.predict_proba(X)[0, 1])

        # Demonstration policy only. Production thresholds require validated
        # data, risk appetite, economics, compliance and human governance.
        if probability < review_threshold:
            decision = "recommend_approve"
        else:
            decision = "review"

        return {
            "default_probability": round(probability, 6),
            "risk_score": round(1000 * (1 - probability)),
            "decision": decision,
            "reason_codes": [],
            "model_version": "demo-catboost-v1",
            "is_synthetic_demo": True,
        }


if __name__ == "__main__":
    scorer = CreditScorer()
    example = {
        "amount": 2_000_000,
        "term_months": 12,
        "revenue": 8_000_000,
        "net_profit": 900_000,
        "employees": 12,
        "collateral": True,
        "product": "working_capital",
        "legal_form": "LLC",
        "profit_margin": 0.1125,
        "business_age_months": 48,
        "payment_to_monthly_revenue": 0.35,
        "revenue_discrepancy": None,
        "debt_load": None,
        "overdue_days": 0,
        "credit_history_months": 36,
        "tax_debt": None,
        "mass_address": False,
        "industry": "retail",
        "unreliable": False,
        "active": True,
    }
    print(scorer.score(example))
