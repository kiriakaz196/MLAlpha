# ML credit scoring demo

**Important:** this repository generates synthetic examples and synthetic `default_12m` labels. It is for pipeline and backend integration testing only. Do not use this model to approve or reject real credit applications.

## Setup

Run these commands from this directory:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
```

## Generate mock data

```bash
python generate_data.py --rows 12000 --out data/mock_credit_applications.csv
```

`default_12m` is the target label. The generator's outcome mechanism is intentionally artificial and should not be interpreted as a validated credit-risk model.

## Train

```bash
python train.py --data data/mock_credit_applications.csv --out-dir artifacts
```

Outputs:
- `artifacts/credit_default_model.cbm`
- `artifacts/metrics.json`
- `artifacts/feature_schema.json`
- `artifacts/feature_importance.csv`

The demo uses a random stratified split because generated rows have no real timeline. For real data, use a time-based split and ensure every feature is known at the decision date.

## Score one application

```bash
python scoring.py
```

Unknown numeric values remain `NaN`; CatBoost supports numeric missing values. Unknown categorical/boolean values are mapped to the explicit category `__MISSING__`, rather than being treated as zero or false.

## FastAPI integration

`fastapi_example.py` illustrates an endpoint. Include its router in your existing app:

```python
from fastapi_example import router as ml_router
app.include_router(ml_router)
```

Adapt the import paths and request schema to your project's layout. Ensure the model artifact exists before starting the app.

## Production checklist

Before any real lending use:
1. Replace mock data and labels with correctly linked historical applications and observed 12-month outcomes.
2. Define default consistently (e.g. 90+ days past due, restructuring, write-off) with risk/compliance stakeholders.
3. Avoid outcome leakage and split by time, not random rows.
4. Evaluate ROC-AUC, PR-AUC, calibration/Brier score, recall at an operational threshold, and segment stability.
5. Calibrate predicted probabilities and set cutoffs based on lender policy and expected loss—not arbitrary demo thresholds.
6. Review fairness, explainability, data quality, security, privacy and applicable laws.
7. Return an explicit `review` outcome for insufficient or conflicting information; do not automatically equate model probability with a lending decision.
