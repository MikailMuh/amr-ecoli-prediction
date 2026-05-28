"""Live prediction interface for user-supplied genomes."""
import io
import tempfile
from pathlib import Path

import streamlit as st
from Bio import SeqIO

from src.features import extract_compositional_features
from src.models import AMRPredictor
from src.visualization import plot_resistance_probability


@st.cache_resource
def load_predictor():
    from src.config import ARTIFACTS_DIR
    missing = [
        f for f in ["lr_v2_models.pkl", "scaler.pkl", "feature_names.pkl"]
        if not (ARTIFACTS_DIR / f).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Model artifacts not found: {missing}. "
            "Run `python scripts/train.py` first."
        )
    return AMRPredictor(model_name="lr_v2")


st.title("Predict Antibiotic Resistance")
st.markdown(
    "Upload a FASTA file of an *E. coli* genome to predict resistance "
    "across nine antibiotics."
)

uploaded = st.file_uploader("FASTA file (.fna, .fasta)", type=["fna", "fasta", "txt"])

if uploaded is None:
    st.info("Awaiting FASTA upload.")
    st.stop()

with tempfile.NamedTemporaryFile(suffix=".fna", delete=False) as tmp:
    tmp.write(uploaded.getvalue())
    fasta_path = Path(tmp.name)

try:
    predictor = load_predictor()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

with st.spinner("Extracting features and running model..."):
    features = extract_compositional_features(fasta_path)
    # BLAST gene features zeroed — BLAST not available in web app
    for col in ["amr_class_beta-lactam", "amr_class_aminoglycoside",
                "amr_class_tetracycline", "amr_class_sulphonamide",
                "amr_class_macrolide", "amr_class_trimethoprim",
                "amr_class_phenicol", "amr_class_quinolone",
                "amr_class_colistin", "amr_class_fosfomycin"]:
        features[col] = 0
    predictions = predictor.predict(features)

st.subheader("Genome Summary")
col1, col2, col3 = st.columns(3)
col1.metric("GC content", f"{features['gc_content']:.3f}")
col2.metric("Genome length", f"{features['valid_length']:,} bp")
col3.metric("Contigs", features["n_contigs"])

st.subheader("Resistance Predictions")
st.dataframe(predictions, use_container_width=True)

st.subheader("Probability Distribution")
fig = plot_resistance_probability(predictions).figure
st.pyplot(fig)

st.caption(
    "Disclaimer: results are for research purposes only and do not constitute "
    "clinical diagnostic guidance."
)