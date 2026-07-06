# CardioRisk AI — Cardiac Arrest Risk Predictor

A Streamlit health informatics app that predicts cardiac arrest probability
using machine learning trained on patient clinical data.

## Modules
- **Risk Prediction** — enter patient characteristics, get instant risk %
- **Upload Patient Data** — upload CSV, model retrains automatically
- **Model Performance** — ROC curve, confusion matrix, AUC metrics
- **About** — feature descriptions and dataset sources

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data format
Download the template from the Upload page, or use:
https://www.kaggle.com/datasets/ronitf/heart-disease-uci
