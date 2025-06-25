import pandas as pd
import os
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer

# === Load real dataset ===
real_data_path = os.path.join('data', 'final_cleaned_drone_dataa.csv')
df = pd.read_csv(real_data_path)

print(f"[INFO] Real dataset shape: {df.shape}")
print(f"[INFO] Columns:\n{df.columns.tolist()}")
print(f"[INFO] Data types:\n{df.dtypes}")

# === Create metadata for the full table ===
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=df)

print("[INFO] Detected metadata:")
print(metadata.to_dict())

# === Train CTGAN on all columns ===
synthesizer = CTGANSynthesizer(metadata)
print("[INFO] Training CTGAN on full dataset...")
synthesizer.fit(df)
print("[INFO] Training complete.")

# === Generate 3000 rows of full synthetic data ===
print("[INFO] Generating synthetic data (3000 rows)...")
synthetic_data = synthesizer.sample(num_rows=3000)
print(f"[INFO] Generated {len(synthetic_data)} synthetic rows.")

# === Save to data folder ===
output_path = os.path.join('data', 'synthetic_data_ctgan_full.csv')
synthetic_data.to_csv(output_path, index=False)
print(f"[INFO] Synthetic data saved to: {output_path}")

# === Optional: Print data summary ===
print("\n[INFO] Real data summary:")
print(df.describe(include='all', datetime_is_numeric=True))
print("\n[INFO] Synthetic data summary:")
print(synthetic_data.describe(include='all', datetime_is_numeric=True))
