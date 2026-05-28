"""Configuration constants for the AMR prediction pipeline."""
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
DATA_DIR = ROOT_DIR / "data"

# Feature extraction
K_MER_SIZES = [3, 4]
MIN_GENOME_LENGTH = 1_500_000

# BLAST parameters
BLAST_MIN_IDENTITY = 90
BLAST_MAX_EVALUE = 1e-50
BLAST_MIN_ALIGN_LENGTH = 200

# Antibiotics in scope
ANTIBIOTICS = [
    "ampicillin",
    "amoxicillin/clavulanic acid",
    "cefotaxime",
    "ceftazidime",
    "cefuroxime",
    "ciprofloxacin",
    "gentamicin",
    "piperacillin/tazobactam",
    "meropenem",
]

# Model hyperparameters
LOGISTIC_REGRESSION_PARAMS = {
    "C": 0.5,
    "max_iter": 2000,
    "class_weight": "balanced",
    "random_state": 42,
}

XGBOOST_PARAMS = {
    "n_estimators": 400,
    "max_depth": 5,
    "learning_rate": 0.05,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
}

RANDOM_FOREST_PARAMS = {
    "n_estimators": 400,
    "max_depth": 12,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "random_state": 42,
}