# Healthcare Risk Prediction & Medical Text Analyzer

An educational portfolio project combining Python, Machine Learning, and rule-based NLP in a Streamlit application. Deep learning and biomedical transformer models are planned as a later extension after the baseline is evaluated.

## Features
- Train and compare Logistic Regression and Random Forest on a binary diabetes health-indicators dataset.
- Evaluate on a held-out test set with accuracy, precision, recall, F1 and ROC-AUC.
- Enter dataset-encoded features in a Streamlit prediction UI.
- Inspect medical text with a small rule-based term dictionary and extractive summary.
- Download evaluation metrics and text-analysis output.

## Dataset setup
1. Download the binary dataset from https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset
2. Place `diabetes_binary_health_indicators_BRFSS2015.csv` inside the `data/` folder.
3. Do not upload private patient records or identifiable clinical notes.

## Run in VS Code (Windows)
Open the project folder in VS Code, then use the terminal:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

If PowerShell blocks environment activation, use Command Prompt:
```bat
.venv\\Scripts\\activate.bat
```

## Limitations and safe use
- This is a learning/demo project, not a medical device or clinically validated risk score.
- Public health-survey labels and model probabilities are not equivalent to an individual's diagnosis or calibrated clinical risk.
- The text analyzer uses keyword matching, not a validated clinical NLP model. It can miss terms, misunderstand negation, and fail on abbreviations.
- Never use the app to diagnose, choose medication, or delay urgent care.
- For future transformer experiments, use properly licensed datasets and de-identified text; follow dataset access requirements.
