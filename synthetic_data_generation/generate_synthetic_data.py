import pandas as pd
from scipy.stats import expon
import numpy as np
import random

# Load original dataset
real_df = pd.read_csv('./data/final_cleaned_drone_dataa.csv')

# Ensure numeric types
real_df["People_Killed"] = pd.to_numeric(real_df["People_Killed"], errors='coerce')
real_df["Min_Civilians"] = pd.to_numeric(real_df["Min_Civilians"], errors='coerce')
real_df["Max_Civilians"] = pd.to_numeric(real_df["Max_Civilians"], errors='coerce')

# Drop rows with NaNs in target columns
real_df = real_df.dropna(subset=["People_Killed", "Min_Civilians", "Max_Civilians"])

# Fit distributions
params_people = expon.fit(real_df["People_Killed"][real_df["People_Killed"] > 0])
params_min = expon.fit(real_df["Min_Civilians"][real_df["Min_Civilians"] > 0])
params_max = expon.fit(real_df["Max_Civilians"][real_df["Max_Civilians"] > 0])

# How many rows you want
num_samples = 4000

# Generate synthetic values
synthetic_people = expon.rvs(*params_people, size=num_samples).round().astype(int).clip(0, 100)
synthetic_min = expon.rvs(*params_min, size=num_samples).round().astype(int).clip(0, 50)
synthetic_max = expon.rvs(*params_max, size=num_samples).round().astype(int).clip(0, 80)

# Check what columns are available and filter out non-existent ones
print("Available columns in the dataset:")
print(real_df.columns.tolist())

# Generate other columns by random sampling from real data
desired_columns = ['Date', 'Location', 'Suspected_Target', 'Latitude', 'Longitude', 'Country']
columns_to_sample = [col for col in desired_columns if col in real_df.columns]

print(f"\nColumns that will be sampled: {columns_to_sample}")

synthetic_other = {}

# Fixed sampling approach using pandas
for col in columns_to_sample:
    clean_values = real_df[col].dropna()
    if len(clean_values) > 0:  # Check if there are any non-null values
        sampled_values = clean_values.sample(n=num_samples, replace=True).tolist()
        synthetic_other[col] = sampled_values
    else:
        print(f"Warning: Column '{col}' has no non-null values, skipping...")
        continue

# Build final synthetic dataframe
synthetic_df = pd.DataFrame(synthetic_other)
synthetic_df["People_Killed"] = synthetic_people
synthetic_df["Min_Civilians"] = synthetic_min
synthetic_df["Max_Civilians"] = synthetic_max

# Reorder columns to match the desired order from your Excel image
desired_column_order = ['Date', 'Location', 'Suspected_Target', 'People_Killed', 'Min_Civilians', 'Max_Civilians', 'Latitude', 'Longitude', 'Country']

# Only include columns that actually exist in the dataframe
final_column_order = [col for col in desired_column_order if col in synthetic_df.columns]
synthetic_df = synthetic_df[final_column_order]

print(f"Final column order: {synthetic_df.columns.tolist()}")

# Save the final synthetic dataset
synthetic_df.to_csv('./data/synthetic_data_4000_full.csv', index=False)
print("Synthetic dataset with 4000 rows saved as 'synthetic_data_4000_full.csv'")