================================================================================
                      CREDIT SCORING MODEL — CodeAlpha Internship
                           ML Project | Dev Shah | 2026
================================================================================

DESCRIPTION
-----------
A machine learning project that predicts whether a person is creditworthy
using the German Credit Dataset. Compares three models — Logistic Regression,
Decision Tree, and Random Forest — with full evaluation metrics, confusion
matrix, ROC curves, and feature importance charts.

Two versions are included:
  • credit_scoring_model.py  → Full production version (auto-downloads dataset)
  • Untitled-1.py            → Local CSV version (manual dataset required)


PROJECT STRUCTURE
-----------------
credit-scoring/
│
├── credit_scoring_model.py   → Main script (fetches dataset via ucimlrepo)
├── Untitled-1.py             → Alternate script (uses local CSV file)
├── german_credit_data.csv    → Required only for Untitled-1.py
│
└── Output Files (auto-generated after running):
    ├── class_distribution.png  → Bar chart of creditworthy vs not
    ├── confusion_matrix.png    → Confusion matrix heatmap (Random Forest)
    ├── roc_curves.png          → ROC curve comparison of all 3 models
    └── feature_importance.png  → Top 10 features (Random Forest)


REQUIREMENTS
------------
Python 3.8+

Install all dependencies:

    pip install pandas numpy matplotlib seaborn scikit-learn ucimlrepo


HOW TO RUN
----------

--- VERSION 1: credit_scoring_model.py (Recommended) ---

    python credit_scoring_model.py

  → Automatically downloads the German Credit Dataset (UCI ID: 144)
  → No CSV file needed
  → Requires internet connection on first run

--- VERSION 2: Untitled-1.py (Local CSV) ---

  1. Place german_credit_data.csv in the same folder as the script
  2. Run:

    python Untitled-1.py

  → Uses local CSV file
  → Works offline


WHAT EACH SCRIPT DOES
----------------------
Step 1  → Loads dataset (auto or CSV)
Step 2  → Cleans and encodes categorical columns
Step 3  → Scales features using StandardScaler
Step 4  → Splits data: 80% train / 20% test
Step 5  → Trains 3 models:
            • Logistic Regression
            • Decision Tree (max_depth=5)
            • Random Forest (100 trees)
Step 6  → Evaluates: Accuracy, Precision, Recall, F1, ROC-AUC
Step 7  → Saves confusion matrix, ROC curves, feature importance charts


TARGET COLUMN
-------------
credit_scoring_model.py:
  → Uses original UCI target (1 = Creditworthy, 2 = Not Creditworthy)
  → Remapped to: 1 = Creditworthy, 0 = Not Creditworthy

Untitled-1.py:
  → Generates target from Credit amount column
  → Credit amount > 5000 → High Risk (1), else Low Risk (0)


MODELS & EXPECTED RESULTS
--------------------------
Model                   Accuracy    ROC-AUC
────────────────────────────────────────────
Logistic Regression     ~74-77%     ~0.79
Decision Tree           ~70-74%     ~0.70
Random Forest           ~78-82%     ~0.85   ← Best Model

Note: Results may vary slightly based on dataset version and random state.


KNOWN BUG IN Untitled-1.py
----------------------------
The original Untitled-1.py has a variable reuse bug:

    y_pred_log = log_model.predict(X_test)
    y_pred_log = dt_model.predict(X_test)   ← overwrites LR prediction
    y_pred_log = rf_model.predict(X_test)   ← overwrites DT prediction

All three print statements end up using only the Random Forest prediction.
Fix: rename variables to y_pred_lr, y_pred_dt, y_pred_rf respectively.


OUTPUT FILES
------------
class_distribution.png   → Shows class imbalance (700 good vs 300 bad)
confusion_matrix.png      → TP / FP / FN / TN breakdown for Random Forest
roc_curves.png            → AUC comparison of all 3 models on one chart
feature_importance.png    → Top 10 most influential features


DATASET INFO
------------
Name     : Statlog (German Credit Data)
Source   : UCI Machine Learning Repository (ID: 144)
Rows     : 1000
Features : 20 (mix of numerical and categorical)
Target   : Creditworthiness (Good / Bad)
Download : https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data


AUTHOR
------
Dev Shah
BCA Semester 2 | YCMOU, Nashik
GitHub : github.com/DevSHah1192
Project: CodeAlpha ML Internship Task

================================================================================
