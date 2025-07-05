# model_new/randomforest_train.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from imblearn.over_sampling import SMOTE

# === Load and preprocess data ===
df = pd.read_csv(r"../data/merged_dataset_ctgan.csv")
numeric_columns = ['Min_Civilians', 'Max_Civilians', 'People_Killed', 'Latitude', 'Longitude']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

df['Civilian_Killed'] = df['Min_Civilians'].apply(lambda x: 1 if x > 0 else 0)
df.drop(columns=['Min_Civilians', 'Max_Civilians', 'Date', 'Location'], inplace=True)
df_encoded = pd.get_dummies(df)

X = df_encoded.drop(columns=['Civilian_Killed'])
y = df_encoded['Civilian_Killed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Apply SMOTE ===
X_res, y_res = SMOTE(random_state=42).fit_resample(X_train, y_train)

# === Train Random Forest ===
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_res, y_res)

# === Evaluate ===
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
cm = confusion_matrix(y_test, y_pred)

print("\n📊 Random Forest Classification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f"ROC AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

# === Save confusion matrix plot ===
os.makedirs("../model/plots", exist_ok=True)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Random Forest with SMOTE")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("../model/plots/confusion_matrix_random_forest_with_smote.png")
plt.close()

# === Save model ===
os.makedirs("../model", exist_ok=True)
joblib.dump(model, "../model/random_forest_model.pkl")
print("\n✅ Random Forest model saved to '../model/'")
