"""Model loading and prediction interface."""
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import ANTIBIOTICS, ARTIFACTS_DIR


class AMRPredictor:
    """Multi-label AMR resistance predictor for E. coli."""

    def __init__(self, model_name: str = "lr_v2"):
        self.model_name = model_name
        self._models: Dict = {}
        self._scaler: StandardScaler = None
        self._feature_names: List[str] = []
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        artifacts_path = ARTIFACTS_DIR
        required = [
            artifacts_path / f"{self.model_name}_models.pkl",
            artifacts_path / "scaler.pkl",
            artifacts_path / "feature_names.pkl",
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing artifacts: {missing}. Run `python scripts/train.py` to generate them."
            )
        with open(artifacts_path / f"{self.model_name}_models.pkl", "rb") as f:
            self._models = pickle.load(f)
        with open(artifacts_path / "scaler.pkl", "rb") as f:
            self._scaler = pickle.load(f)
        with open(artifacts_path / "feature_names.pkl", "rb") as f:
            self._feature_names = pickle.load(f)

    def predict(self, features: Dict[str, float]) -> pd.DataFrame:
        """Predict resistance for all antibiotics from extracted features."""
        feature_vector = pd.DataFrame([features])[self._feature_names]
        scaled = self._scaler.transform(feature_vector)

        results = []
        for antibiotic in ANTIBIOTICS:
            if antibiotic not in self._models:
                continue
            model = self._models[antibiotic]
            prediction = int(model.predict(scaled)[0])
            probability = float(model.predict_proba(scaled)[0, 1])
            results.append({
                "antibiotic": antibiotic,
                "prediction": "Resistant" if prediction == 1 else "Susceptible",
                "probability": probability,
            })

        return pd.DataFrame(results)

    @property
    def feature_names(self) -> List[str]:
        return self._feature_names