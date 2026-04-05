import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder , StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import( accuracy_score , classification_report , confusion_matrix,
                             precision_recall_curve,recall_score, roc_curve,
                             f1_score,roc_auc_score)
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("german_credit_data.csv")

df.drop(columns=["Unnamed: 0"],inplace=True)
le=LabelEncoder()
scaler = StandardScaler()
cols=[
    "Sex",
    "Housing",
    "Saving accounts",
    "Checking account",
    "Purpose"
]
for col in cols:
    df[col]=le.fit_transform(df[col].astype(str))
    
    df["CreditRisk"]=(df["Credit amount"]>5000).astype(int)
    y = df["CreditRisk"]
    X = df.drop(["CreditRisk", "Credit amount"], axis=1)
print(df.head())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
X_scaled, y, test_size=0.2, random_state=42
)

log_model = LogisticRegression()
log_model.fit(X_train,y_train)

dt_model = DecisionTreeClassifier()
dt_model.fit(X_train,y_train)

rf_model = RandomForestClassifier()
rf_model.fit(X_train,y_train)


y_pred_log = log_model.predict(X_test)
y_pred_log = dt_model.predict(X_test)
y_pred_log = rf_model.predict(X_test)

print("Logistic Regression Accuracy ",accuracy_score (y_test,y_pred_log))
print(" Decision Tree Accuracy ",accuracy_score(y_test,y_pred_log))
print(" Random Forest Classifier ",accuracy_score(y_test,y_pred_log))

print("\n Logistic Regression ")
print(classification_report(y_test,y_pred_log))

print("\n  Decision Tree ")
print(classification_report(y_test,y_pred_log))
 
print(" Random Forest Classifier")
print(classification_report(y_test,y_pred_log))


cm = confusion_matrix(y_test,y_pred_log)

sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
plt.xlabel("Prediction")
plt.ylabel("Actual")
plt.title("Random Forest Confusion Matrix")
plt.show()

y_prob = rf_model.predict_proba(X_test)[:, 1]
fpr,tpr , threesolds = roc_curve(y_test,y_prob)

plt.plot(fpr,tpr)
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate ")
plt.ylabel("True Positive Rate ")
plt.title("Roc Curve")
plt.show()

print("ROC Acu Score:",roc_auc_score(y_test,y_prob))

importances = rf_model.feature_importances_

feat_importance = pd.Series(importances, index=X.columns)

feat_importance.sort_values().plot(kind='barh')
plt.title("Feature Importance")
plt.show()