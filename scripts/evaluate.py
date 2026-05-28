"""Evaluate trained AMR classifiers on test set.

Usage:
    python scripts/evaluate.py
"""
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, roc_auc_score, matthews_corrcoef,
    precision_score, recall_score,
)

from src.config import ANTIBIOTICS, ARTIFACTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_test_split():
    with np.load(ARTIFACTS_DIR / "test_split.npz", allow_pickle=True) as data:
        return data["X_test"].copy(), data["y_test"].copy(), data["test_index"].copy()


def load_models(filename):
    with open(ARTIFACTS_DIR / filename, "rb") as f:
        return pickle.load(f)


def evaluate_model(models, model_name, X_test, y_test_df):
    results = []
    for i, antibiotic in enumerate(ANTIBIOTICS):
        if antibiotic not in models:
            continue
        y_true = y_test_df[:, i]
        mask = ~np.isnan(y_true)
        y_true_clean = y_true[mask].astype(int)
        X_clean = X_test[mask]

        if len(np.unique(y_true_clean)) < 2:
            continue

        model = models[antibiotic]
        y_pred = model.predict(X_clean)
        y_proba = model.predict_proba(X_clean)[:, 1]

        results.append({
            "model": model_name,
            "antibiotic": antibiotic,
            "F1": f1_score(y_true_clean, y_pred, zero_division=0),
            "AUC": roc_auc_score(y_true_clean, y_proba),
            "MCC": matthews_corrcoef(y_true_clean, y_pred),
            "Precision": precision_score(y_true_clean, y_pred, zero_division=0),
            "Recall": recall_score(y_true_clean, y_pred, zero_division=0),
            "n_test": int(mask.sum()),
            "n_positive": int(y_true_clean.sum()),
        })

    return pd.DataFrame(results)


def main():
    X_test, y_test, _ = load_test_split()
    logger.info("Loaded test set: %d samples", len(X_test))

    all_results = []
    for filename, name in [
        ("lr_v2_models.pkl", "LogisticRegression"),
        ("rf_v2_models.pkl", "RandomForest"),
        ("xgb_v2_models.pkl", "XGBoost"),
    ]:
        if not (ARTIFACTS_DIR / filename).exists():
            logger.warning("Skipping %s (not found)", filename)
            continue
        models = load_models(filename)
        results = evaluate_model(models, name, X_test, y_test)
        all_results.append(results)
        logger.info("%s macro AUC: %.3f", name, results["AUC"].mean())

    combined = pd.concat(all_results, ignore_index=True)
    output_path = ARTIFACTS_DIR / "cv_results.csv"
    combined.to_csv(output_path, index=False)
    logger.info("Saved evaluation results to %s", output_path)

    print("\nMacro performance:")
    print(combined.groupby("model")[["F1", "AUC", "MCC"]].mean().round(3))


if __name__ == "__main__":
    main()