import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# 1. Load the dataset
file_path = r"C:\Users\Kushagra Sharma\Downloads\merged_dataset (1).csv"
df = pd.read_csv(file_path)

# 2. Convert numeric columns
numeric_columns = ['Min_Civilians', 'Max_Civilians', 'People_Killed', 'Latitude', 'Longitude']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# 3. Create binary target
df['Civilian_Killed'] = df['Min_Civilians'].apply(lambda x: 1 if x > 0 else 0)

# 4. Drop unwanted columns
df.drop(columns=['Min_Civilians', 'Max_Civilians', 'Date', 'Location'], inplace=True)

# 5. Encode categorical features
df_encoded = pd.get_dummies(df)

# 6. Feature matrix and target
X = df_encoded.drop(columns=['Civilian_Killed'])
y = df_encoded['Civilian_Killed']

# 7. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Train models
logreg = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
logreg.fit(X_train, y_train)
rf.fit(X_train, y_train)

# 9. Evaluation function
def evaluate_model(model, X_test, y_test, label):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n📊 {label} Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {label}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

# 10. Evaluate both
evaluate_model(logreg, X_test, y_test, "Logistic Regression")
evaluate_model(rf, X_test, y_test, "Random Forest")

# 11. Save models
os.makedirs("model", exist_ok=True)
joblib.dump(logreg, "model/logistic_regression_model.pkl")
joblib.dump(rf, "model/random_forest_model.pkl")

print("\n✅ Models saved to 'model/' directory.")
