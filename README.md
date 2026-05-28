# AMR E. coli Multi-Label Prediction

Machine learning system for predicting antibiotic resistance phenotypes in
*Escherichia coli* using BioPython-extracted genomic features and BLAST-based
AMR gene detection.

## Live Demo
🔗 [amr-ecoli-prediction.streamlit.app](https://amr-ecoli-prediction.streamlit.app)

## Quick Start

\`\`\`bash
git clone https://github.com/[user]/amr-ecoli-prediction.git
cd amr-ecoli-prediction
pip install -r requirements.txt
streamlit run app.py
\`\`\`

## Methodology
- **Dataset:** BV-BRC E. coli AMR phenotypes (302 genomes)
- **Features:** 328 compositional (GC, k-mer k=3, k=4) + 10 functional (BLAST + ResFinder)
- **Models:** Logistic Regression (best), Random Forest, XGBoost
- **Evaluation:** 5-fold stratified cross-validation
- **Best result:** AUC 0.768 ± 0.039 (Logistic Regression)

## Project Structure
\`\`\`
├── app.py               # Streamlit entry point
├── pages/               # Multi-page Streamlit views
├── src/                 # Core Python modules
│   ├── config.py
│   ├── features.py
│   ├── models.py
│   └── visualization.py
├── scripts/             # Training & evaluation scripts
└── notebooks/           # Original development notebooks
\`\`\`

## Citation
If you use this code, please cite our paper [to be published].

## License
MIT