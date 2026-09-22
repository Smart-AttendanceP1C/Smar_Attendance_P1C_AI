"""
ATT-ML: Logistic regression for next-week low-attendance risk
================================================================
 
Reads weekly_features_and_flags.xlsx (produced by the Data Analyst pod)
and trains a simple logistic regression to predict label_next_week_low.
 
Per the brief's "AI release gate", this script also prints the baseline's
own precision/recall on the exact same test split, so the two numbers are
directly comparable (same rows, same fold).
"""
 
import hashlib
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, brier_score_loss, classification_report,
)

# --- Project layout -------------------------------------------------------
# This file lives at <project_root>/src/train_weekly_risk_model.py.
# Resolving paths from __file__ (not from the current working directory)
# means the script runs the same whether you launch it from the project
# root, from src/, or from anywhere else.
THIS_FILE = Path(__file__).resolve()
SRC_DIR = THIS_FILE.parent
PROJECT_ROOT = SRC_DIR.parent

IN_PATH = PROJECT_ROOT / "demo_data" / "weekly_features_and_flags.xlsx"
MODEL_PATH = PROJECT_ROOT / "models" / "weekly_risk_model.joblib"
MODEL_HASH_PATH = PROJECT_ROOT / "models" / "weekly_risk_model.sha256"  # integrity manifest for the API to check
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)  # ensure models/ exists
 
FEATURE_COLS = [
    "attendance_rate_to_date",
    "trend_last_3",
    "rejected_last_2",
    "consecutive_absences",
    "course_load",
]
LABEL_COL = "label_next_week_low"
BASELINE_FLAG_COL = "flag_any"  # the rule-based baseline's own prediction
 
RANDOM_STATE = 42
 
# ---------------------------------------------------------------------------
# Load + clean
# ---------------------------------------------------------------------------
df = pd.read_excel(IN_PATH, sheet_name="WeeklyFeaturesAndFlags")
df = df.dropna(subset=[LABEL_COL]).copy()
df = df.dropna(subset=FEATURE_COLS).copy()
 
X = df[FEATURE_COLS]
y = df[LABEL_COL].astype(int)
baseline_pred = df[BASELINE_FLAG_COL].astype(int)
 
print(f"Rows used for train+test: {len(df)}")
print(f"Positive rate (label_next_week_low=1): {y.mean():.3f}")
 
# ---------------------------------------------------------------------------
# Documented train/test split
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test, baseline_train, baseline_test = train_test_split(
    X, y, baseline_pred,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=y,
)
print(f"Train rows: {len(X_train)}   Test rows: {len(X_test)}")
 
# ---------------------------------------------------------------------------
# Train logistic regression
# ---------------------------------------------------------------------------
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)
 
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)
 
# Persist the model so weekly_risk_api.py can load it without retraining.
joblib.dump(model, MODEL_PATH)
print(f"Model saved: {MODEL_PATH}")
 
# Security Finding 2 fix: write a SHA256 hash of the model file so the API
# can verify the file hasn't been swapped/tampered with before loading it.
model_bytes = open(MODEL_PATH, "rb").read()
model_hash = hashlib.sha256(model_bytes).hexdigest()
with open(MODEL_HASH_PATH, "w") as f:
    f.write(model_hash)
print(f"Model hash saved: {MODEL_HASH_PATH} ({model_hash[:16]}...)")
 
# ---------------------------------------------------------------------------
# Evaluate: model vs. baseline, same test rows
# ---------------------------------------------------------------------------
def report(name, y_true, y_hat, y_probability=None):
    p = precision_score(y_true, y_hat, zero_division=0)
    r = recall_score(y_true, y_hat, zero_division=0)
    f1 = f1_score(y_true, y_hat, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat).ravel()
    print(f"\n--- {name} ---")
    print(f"  precision={p:.3f}  recall={r:.3f}  f1={f1:.3f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    if y_probability is not None:
        brier = brier_score_loss(y_true, y_probability)
        print(f"  Brier score (calibration, lower=better): {brier:.3f}")
 
report("Rule-based BASELINE (on test rows)", y_test, baseline_test)
report("Logistic Regression MODEL", y_test, y_pred, y_prob)
 
print("\nFull classification report for the model:")
print(classification_report(y_test, y_pred, digits=3))
 
coefs = pd.Series(model.coef_[0], index=FEATURE_COLS).sort_values()
print("\nModel coefficients (positive = raises risk, negative = lowers it):")
print(coefs.to_string())
 
calib_df = pd.DataFrame({"y_true": y_test.values, "y_prob": y_prob})
calib_df["risk_band"] = pd.cut(
    calib_df["y_prob"], bins=[0, 0.25, 0.5, 0.75, 1.0],
    labels=["low", "medium", "high", "very_high"],
)
calib_table = calib_df.groupby("risk_band", observed=True).agg(
    n=("y_true", "size"),
    predicted_avg_prob=("y_prob", "mean"),
    actual_rate=("y_true", "mean"),
)
print("\nCalibration by risk band (predicted_avg_prob should track actual_rate):")
print(calib_table.to_string())