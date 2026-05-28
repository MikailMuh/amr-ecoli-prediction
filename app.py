"""Streamlit application entry point."""
import streamlit as st

st.set_page_config(
    page_title="AMR E. coli Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("E. coli Antibiotic Resistance Predictor")
st.markdown(
    """
    A multi-label machine learning system for predicting antibiotic resistance
    in *Escherichia coli* using genomic features. This research compares Logistic
    Regression, Random Forest, and XGBoost classifiers on a curated dataset of
    302 *E. coli* genomes from the BV-BRC repository.

    Navigate using the sidebar to explore results, run predictions, or view
    model interpretability analyses.
    """
)

st.subheader("Project Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Genomes analyzed", "302")
col2.metric("Antibiotics", "9")
col3.metric("Best AUC-ROC", "0.768")

st.subheader("Citation")
st.code(
    "[Authors]. Multi-Label Antibiotic Resistance Prediction in E. coli "
    "Using Machine Learning and BioPython-Extracted Genomic Features. "
    "IEEE Conference, 2026.",
    language="text",
)