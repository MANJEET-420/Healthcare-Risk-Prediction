from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

# Recommended Kaggle file:
# diabetes_binary_health_indicators_BRFSS2015.csv
csv_files = list(DATA_DIR.glob("*.csv"))
if not csv_files:
    raise FileNotFoundError(
        "data/ folder mein diabetes CSV nahi mili. README.md ke dataset steps follow karo."
    )

# Prefer the binary dataset when multiple CSV files are present.
preferred = [p for p in csv_files if "binary_health_indicators" in p.name.lower()
             and "5050" not in p.name.lower()]
csv_path = preferred[0] if preferred else csv_files[0]
print(f"Dataset loaded: {csv_path.name}")

df = pd.read_csv(csv_path)
df.columns = [str(c).strip() for c in df.columns]

target_candidates = ["Diabetes_binary", "diabetes_binary", "Diabetes_012", "target", "label"]
target_col = next((c for c in target_candidates if c in df.columns), None)
if target_col is None:
    raise ValueError(
        f"Target column nahi mili. Available columns: {list(df.columns)}. "
        "Binary dataset use karo jisme target 'Diabetes_binary' ho."
    )

# This starter intentionally supports binary classification only.
y = df[target_col]
unique_values = sorted(pd.Series(y.dropna().unique()).tolist())
if not set(unique_values).issubset({0, 1}):
    raise ValueError(
        f"Target '{target_col}' binary 0/1 nahi hai (values: {unique_values}). "
        "Kaggle se diabetes_binary_health_indicators_BRFSS2015.csv use karo."
    )

X = df.drop(columns=[target_col]).select_dtypes(include="number")
if X.empty:
    raise ValueError("Dataset mein numeric feature columns nahi mile.")

# Remove rows where the target is missing; impute missing feature values inside pipeline.
valid = y.notna()
X, y = X.loc[valid], y.loc[valid].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ]),
    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=150, random_state=42, class_weight="balanced_subsample",
            n_jobs=-1, min_samples_leaf=2
        ))
    ])
}

results = {}
best_name, best_model, best_auc = None, None, -1

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probabilities) if y_test.nunique() == 2 else float("nan")
    results[name] = {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
        "roc_auc": round(float(auc), 4),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "classification_report": classification_report(
            y_test, pred, zero_division=0, output_dict=True
        )
    }
    print(f"\n{name}: {results[name]}")
    if auc > best_auc:
        best_name, best_model, best_auc = name, model, auc

joblib.dump(best_model, ARTIFACTS_DIR / "risk_model.joblib")
with open(ARTIFACTS_DIR / "feature_names.json", "w", encoding="utf-8") as f:
    json.dump(list(X.columns), f, indent=2)
with open(ARTIFACTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump({"best_model": best_name, "results": results, "target": target_col}, f, indent=2)

print(f"\nBest model by ROC-AUC: {best_name}")
print("Saved artifacts/risk_model.joblib, feature_names.json, metrics.json")
print("Reminder: this is a learning model, not a clinically validated diagnostic tool.")
