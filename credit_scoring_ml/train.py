"""Train and evaluate a synthetic/demo CatBoost credit-risk model."""
from __future__ import annotations
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split

try:
    from .features import FEATURES, MODEL_CATEGORICAL_FEATURES, prepare_features
except ImportError:  # running this file directly from the ml directory
    from features import FEATURES, MODEL_CATEGORICAL_FEATURES, prepare_features

TARGET = "default_12m"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/mock_credit_applications.csv")
    parser.add_argument("--out-dir", default="artifacts")
    parser.add_argument("--iterations", type=int, default=700)
    args = parser.parse_args()

    data_path = Path(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    if TARGET not in df.columns:
        raise ValueError(f"Dataset must contain target column {TARGET!r}")
    df = df[df[TARGET].notna()].copy()
    y = df[TARGET].astype(int)
    if not set(y.unique()).issubset({0, 1}) or y.nunique() < 2:
        raise ValueError("default_12m must contain both binary classes 0 and 1")

    X = prepare_features(df)
    # 60% train, 20% validation, 20% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.40,
        random_state=42,
        stratify=y,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp,
    )

    cat_indices = [
        X.columns.get_loc(c)
        for c in MODEL_CATEGORICAL_FEATURES
    ]

    model = CatBoostClassifier(
        iterations=1500,
        depth=6,
        learning_rate=0.03,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=100,
        allow_writing_files=False,
        # Для оценки вероятностей начнём без балансировки весами.
    )

    model.fit(
        X_train,
        y_train,
        cat_features=cat_indices,
        eval_set=(X_val, y_val),
        early_stopping_rounds=100,
        verbose=100,
    )

    # Только теперь оцениваем на тесте
    probabilities = model.predict_proba(X_test)[:, 1]

    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        brier_score_loss,
        recall_score,
        precision_score,
    )

    from sklearn.metrics import f1_score

    print("\nThreshold analysis:")
    print(
        f"{'Threshold':>10} | {'Precision':>9} | {'Recall':>7} | "
        f"{'F1':>6} | {'FN':>6} | {'FP':>6}"
    )
    print("-" * 62)

    for threshold in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
        threshold_predictions = (probabilities >= threshold).astype(int)

        precision = precision_score(
            y_test, threshold_predictions, zero_division=0
        )
        recall = recall_score(
            y_test, threshold_predictions, zero_division=0
        )
        f1 = f1_score(
            y_test, threshold_predictions, zero_division=0
        )

        tn, fp, fn, tp = confusion_matrix(
            y_test, threshold_predictions, labels=[0, 1]
        ).ravel()

        print(
            f"{threshold:10.2f} | {precision:9.3f} | {recall:7.3f} | "
            f"{f1:6.3f} | {fn:6d} | {fp:6d}"
        )

    predictions = (probabilities >= 0.5).astype(int)

    print("ROC-AUC:", roc_auc_score(y_test, probabilities))
    print("PR-AUC:", average_precision_score(y_test, probabilities))
    print("Brier score:", brier_score_loss(y_test, probabilities))
    print("Default recall:", recall_score(y_test, predictions))
    print("Default precision:", precision_score(
        y_test, predictions, zero_division=0
    ))

    predicted = (probabilities >= 0.5).astype(int)

    metrics = {
        "dataset": str(data_path),
        "rows": int(len(df)),
        "synthetic_data_warning": True,
        "default_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "brier_score": float(brier_score_loss(y_test, probabilities)),
        "threshold_for_demo_classification": 0.5,
        "confusion_matrix_labels_0_1": confusion_matrix(y_test, predicted).tolist(),
        "classification_report": classification_report(
            y_test, predicted, output_dict=True, zero_division=0
        ),
    }

    model_path = out_dir / "credit_default_model.cbm"
    model.save_model(model_path)
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (out_dir / "feature_schema.json").write_text(
        json.dumps({"features": FEATURES, "categorical_features": MODEL_CATEGORICAL_FEATURES},
                   indent=2), encoding="utf-8"
    )

    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.get_feature_importance()
    }).sort_values("importance", ascending=False)
    importance.to_csv(out_dir / "feature_importance.csv", index=False)

    print("\nEvaluation (synthetic data only):")
    print(json.dumps({k: v for k, v in metrics.items()
                      if k not in ("classification_report", "confusion_matrix_labels_0_1")},
                     indent=2))
    print("\nTop feature importances:")
    print(importance.head(15).to_string(index=False))
    print(f"\nModel saved to: {model_path}")
    print("Do not interpret these metrics as evidence of real-world credit-risk performance.")


if __name__ == "__main__":
    main()
