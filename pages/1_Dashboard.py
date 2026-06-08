"""Results dashboard showing model performance metrics."""
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.config import ARTIFACTS_DIR

st.title("Results Dashboard")

_CV_RESULTS_PATH = ARTIFACTS_DIR / "cv_results.csv"


@st.cache_data
def load_cv_results() -> pd.DataFrame:
    return pd.read_csv(_CV_RESULTS_PATH)


if not _CV_RESULTS_PATH.exists():
    st.warning(
        "Cross-validation results not found. "
        "Run `python scripts/evaluate.py` first to generate `artifacts/cv_results.csv`."
    )
    st.stop()

cv_results = load_cv_results()

st.subheader("Macro Performance (5-fold Cross-Validation)")
macro = cv_results.groupby("model")[["F1", "AUC", "MCC"]].agg(["mean", "std"]).round(3)
st.dataframe(macro, use_container_width=True)

st.subheader("Per-Antibiotic AUC-ROC")
auc_pivot = (
    cv_results.groupby(["model", "antibiotic"])["AUC"]
    .mean()
    .unstack(level=0)
    .round(3)
)
fig, ax = plt.subplots(figsize=(12, 5))
auc_pivot.plot(kind='bar', ax=ax, width=0.8)
ax.set_xlabel('Antibiotic')
ax.set_ylabel('AUC')
ax.set_title('Per-Antibiotic AUC-ROC by Model')
ax.legend(title='Model')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
st.pyplot(fig)

st.subheader("Methodology")
st.markdown(
    """
    **Pipeline:** BioPython-based compositional features (GC content, k-mer
    frequencies for k=3 and k=4) combined with functional features from
    BLAST against the ResFinder database. Total: 338 features per genome.

    **Class imbalance:** Handled via class weighting; SMOTE was evaluated
    but excluded due to degradation in tree-based models.

    **Evaluation:** 5-fold stratified cross-validation using
    MultilabelStratifiedKFold from iterative-stratification.
    """
)
