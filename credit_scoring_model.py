import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, RocCurveDisplay, classification_report
)

import warnings
warnings.filterwarnings("ignore")



print("=" * 55)
print("       CREDIT SCORING MODEL — CodeAlpha")
print("=" * 55)
print("\n📥 Loading German Credit Dataset...")

dataset = fetch_ucirepo(id=144)    
X = dataset.data.features            
y = dataset.data.targets            

y = y.squeeze()                      
y = y.map({1: 1, 2: 0})             # 1=creditworthy, 0=not creditworthy

print(f"✅ Dataset loaded: {X.shape[0]} rows, {X.shape[1]} features")
print(f"   Creditworthy (1): {y.sum()}  |  Not Creditworthy (0): {(y==0).sum()}")



print("\n" + "─" * 55)
print("📊 STEP 3: Exploratory Data Analysis")
print("─" * 55)
print("\nFirst 5 rows of the dataset:")
print(X.head())
print("\nData types and missing values:")
print(X.info())
print("\nBasic statistics:")
print(X.describe())


missing = X.isnull().sum().sum()
print(f"\n❓ Total missing values: {missing}")

plt.figure(figsize=(5, 4))
y.value_counts().plot(kind='bar', color=['steelblue', 'salmon'], edgecolor='black')
plt.xticks([0, 1], ['Not Creditworthy (0)', 'Creditworthy (1)'], rotation=0)
plt.title('Class Distribution')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('class_distribution.png', dpi=150)
plt.show()
print("✅ Class distribution chart saved.")



print("\n" + "─" * 55)
print("🔧 STEP 4: Feature Engineering & Preprocessing")
print("─" * 55)


categorical_cols = X.select_dtypes(include='object').columns.tolist()
numerical_cols   = X.select_dtypes(include=['int64','float64']).columns.tolist()

print(f"\nCategorical columns ({len(categorical_cols)}): {categorical_cols}")
print(f"Numerical columns   ({len(numerical_cols)}):   {numerical_cols}")


le = LabelEncoder()
X_encoded = X.copy()
for col in categorical_cols:
    X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))

print("\n✅ Categorical columns encoded.")


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)
X_scaled = pd.DataFrame(X_scaled, columns=X_encoded.columns)

print("✅ Feature scaling complete.")


print("\n" + "─" * 55)
print("✂️  STEP 5: Train / Test Split")
print("─" * 55)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
# stratify=y ensures both splits have the same class ratio

print(f"\nTraining samples : {X_train.shape[0]}")
print(f"Testing  samples : {X_test.shape[0]}")



print("\n" + "─" * 55)
print("🤖 STEP 6: Training ML Models")
print("─" * 55)


lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)
print("✅ Logistic Regression trained.")


dt_model = DecisionTreeClassifier(random_state=42, max_depth=5)
dt_model.fit(X_train, y_train)
print("✅ Decision Tree trained.")


rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
print("✅ Random Forest trained.")


print("\n" + "─" * 55)
print("📈 STEP 7: Model Evaluation")
print("─" * 55)

models = {
    "Logistic Regression": lr_model,
    "Decision Tree":       dt_model,
    "Random Forest":       rf_model
}

results = []

for name, model in models.items():
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]  # probability of class 1

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_pred_prob)

    results.append({
        "Model": name, "Accuracy": acc, "Precision": prec,
        "Recall": rec, "F1-Score": f1, "ROC-AUC": auc
    })

    print(f"\n{'─'*30}")
    print(f"  {name}")
    print(f"{'─'*30}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")

results_df = pd.DataFrame(results).set_index("Model")
print("\n\n📋 SUMMARY TABLE:")
print(results_df.round(4).to_string())

print("\n" + "─" * 55)
print("🔲 STEP 8: Confusion Matrix — Random Forest")
print("─" * 55)

rf_pred = rf_model.predict(X_test)
cm = confusion_matrix(y_test, rf_pred)

fig, ax = plt.subplots(figsize=(5, 4))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=["Bad Credit (0)", "Good Credit (1)"])
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title("Confusion Matrix — Random Forest")
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()
print("✅ Confusion matrix saved.")
print("\n" + "─" * 55)
print("📉 STEP 9: ROC Curves — All Models")
print("─" * 55)

fig, ax = plt.subplots(figsize=(7, 5))
for name, model in models.items():
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name)
ax.set_title("ROC Curve Comparison")
ax.plot([0, 1], [0, 1], 'k--', label='Random Guess')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('roc_curves.png', dpi=150)
plt.show()
print("✅ ROC curve saved.")

print("\n" + "─" * 55)
print("⭐ STEP 10: Feature Importance — Random Forest")
print("─" * 55)

importances = rf_model.feature_importances_
feat_df = pd.DataFrame({
    'Feature': X_encoded.columns,
    'Importance': importances
}).sort_values('Importance', ascending=False).head(10)

plt.figure(figsize=(8, 5))
sns.barplot(data=feat_df, x='Importance', y='Feature',
            palette='viridis', edgecolor='black')
plt.title("Top 10 Feature Importances — Random Forest")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()
print("✅ Feature importance chart saved.")
print("\nTop 5 most important features:")
print(feat_df[['Feature','Importance']].head(5).to_string(index=False))



print("\n" + "─" * 55)
print("📄 STEP 11: Full Classification Report — Random Forest")
print("─" * 55)
print(classification_report(y_test, rf_pred,
      target_names=["Bad Credit (0)", "Good Credit (1)"]))


print("\n" + "=" * 55)

print("=" * 55)
print("\n Files saved:")
print("   • class_distribution.png")
print("   • confusion_matrix.png")
print("   • roc_curves.png")
print("   • feature_importance.png")


