"""Example integration; adapt imports and response schemas to your backend."""
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict
from typing import Any

try:
    from .scoring import CreditScorer
except ImportError:  # adjust to your backend package layout if needed
    from scoring import CreditScorer

router = APIRouter(prefix="/ml", tags=["credit-scoring"])
scorer = CreditScorer(Path(__file__).parent / "artifacts" / "credit_default_model.cbm")


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    amount: float | None = None
    term_months: int | None = None
    revenue: float | None = None
    net_profit: float | None = None
    employees: int | None = None
    collateral: bool | None = None
    product: str | None = None
    legal_form: str | None = None
    profit_margin: float | None = None
    business_age_months: int | None = None
    payment_to_monthly_revenue: float | None = None
    revenue_discrepancy: float | None = None
    debt_load: float | None = None
    overdue_days: int | None = None
    credit_history_months: int | None = None
    tax_debt: float | None = None
    mass_address: bool | None = None
    industry: str | None = None
    unreliable: bool | None = None
    active: bool | None = None


@router.post("/score")
def score_application(request: ScoreRequest) -> dict[str, Any]:
    try:
        return scorer.score(request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
