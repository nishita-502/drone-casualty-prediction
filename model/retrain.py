import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

file_path = r"C:\Users\Kushagra Sharma\Downloads\merged_dataset (1).csv"
df = pd.read_csv(file_path)

numeric_columns = ['Min_Civilians', 'Max_Civilians', 'People_Killed', 'Latitude', 'Longitude']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

df['Civilian_Killed'] = df['Min_Civilians'].apply(lambda x: 1 if x > 0 else 0)
df.drop(columns=['Min_Civilians', 'Max_Civilians', 'Date', 'Location'], inplace=True)

df_encoded = pd.get_dummies(df)
X = df_encoded.drop(columns=['Civilian_Killed'])
y = df_encoded['Civilian_Killed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

os.makedirs("model", exist_ok=True)
joblib.dump(rf, "model/random_forest_model.pkl")
print("🔁 Model retrained and saved.")
