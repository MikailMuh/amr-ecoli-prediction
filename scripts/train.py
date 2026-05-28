"""Train AMR resistance classifiers and save artifacts.

Usage:
    python scripts/train.py --features data/X_combined.csv --labels data/y_combined.csv
"""
import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import xgboost as xgb

from src.config import (
    ANTIBIOTICS,
    ARTIFACTS_DIR,
    MIN_GENOME_LENGTH,
    LOGISTIC_REGRESSION_PARAMS,
    RANDOM_FOREST_PARAMS,
    XGBOOST_PARAMS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train AMR resistance classifiers")
    parser.add_argument("--features", type=Path, required=True, help="Path to X features CSV")
    parser.add_argument("--labels", type=Path, required=True, help="Path to y labels CSV")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_data(features_path: Path, labels_path: Path):
    X = pd.read_csv(features_path, index_col=0)
    y = pd.read_csv(labels_path, index_col=0)
    X.index = X.index.astype(str)
    y.index = y.index.astype(str)
    return X, y


def apply_length_filter(X: pd.DataFrame, y: pd.DataFrame):
    mask = X["valid_length"] >= MIN_GENOME_LENGTH
    return X[mask], y.loc[X[mask].index]


def train_per_antibiotic(model_factory, X_train, y_train_df, use_pos_weight=False):
    models = {}
    for antibiotic in ANTIBIOTICS:
        mask = y_train_df[antibiotic].notna()
        X_subset = X_train[mask.values]
        y_subset = y_train_df[antibiotic][mask].astype(int).values

        n_pos = int(y_subset.sum())
        n_neg = len(y_subset) - n_pos
        if n_pos < 3 or n_neg < 3:
            logger.warning("Skipping %s (insufficient samples: %d pos, %d neg)",
                          antibiotic, n_pos, n_neg)
            continue

        model = model_factory(n_pos, n_neg) if use_pos_weight else model_factory()
        model.fit(X_subset, y_subset)
        models[antibiotic] = model
        logger.info("Trained %s: %d samples (%d resistant)", antibiotic, len(y_subset), n_pos)

    return models


def main():
    args = parse_args()
    np.random.seed(args.seed)

    logger.info("Loading data from %s", args.features)
    X, y = load_data(args.features, args.labels)
    X, y = apply_length_filter(X, y)
    logger.info("Filtered dataset: %d genomes, %d features", X.shape[0], X.shape[1])

    y_for_split = y.fillna(0).astype(int)
    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=args.test_size, random_state=args.seed
    )
    train_idx, test_idx = next(splitter.split(X, y_for_split))
    X_train_raw, X_test_raw = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    logger.info("Training Logistic Regression")
    lr_models = train_per_antibiotic(
        lambda: LogisticRegression(**LOGISTIC_REGRESSION_PARAMS),
        X_train, y_train,
    )

    logger.info("Training Random Forest")
    rf_models = train_per_antibiotic(
        lambda: RandomForestClassifier(**RANDOM_FOREST_PARAMS),
        X_train, y_train,
    )

    logger.info("Training XGBoost")
    def xgb_factory(n_pos, n_neg):
        params = dict(XGBOOST_PARAMS)
        params["scale_pos_weight"] = n_neg / max(n_pos, 1)
        params["eval_metric"] = "logloss"
        return xgb.XGBClassifier(**params)

    xgb_models = train_per_antibiotic(xgb_factory, X_train, y_train, use_pos_weight=True)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "lr_v2_models.pkl": lr_models,
        "rf_v2_models.pkl": rf_models,
        "xgb_v2_models.pkl": xgb_models,
        "scaler.pkl": scaler,
        "feature_names.pkl": X.columns.tolist(),
        "antibiotics.pkl": ANTIBIOTICS,
    }

    for filename, obj in artifacts.items():
        path = ARTIFACTS_DIR / filename
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        logger.info("Saved %s (%.2f MB)", path, path.stat().st_size / 1e6)

    # Save test split untuk evaluate.py — reorder columns to match ANTIBIOTICS order
    y_test_ordered = y_test.reindex(columns=ANTIBIOTICS)
    np.savez(
        ARTIFACTS_DIR / "test_split.npz",
        X_test=X_test,
        y_test=y_test_ordered.values,
        test_index=y_test.index.values,
    )

    logger.info("Training complete")


if __name__ == "__main__":
    main()