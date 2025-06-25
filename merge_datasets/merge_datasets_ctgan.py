import pandas as pd

# Load original and synthetic datasets
original_df = pd.read_csv('./data/final_cleaned_drone_dataa.csv')
synthetic_df = pd.read_csv('./data/synthetic_data_ctgan_full.csv')

# Optional: Add a column to identify the source
original_df["Source"] = "Real"
synthetic_df["Source"] = "Synthetic"

# Make sure both have the same columns (reorder synthetic if needed)
synthetic_df = synthetic_df[original_df.columns]

# Concatenate both
merged_df = pd.concat([original_df, synthetic_df], ignore_index=True)

# Save the merged dataset
merged_df.to_csv('./data/merged_dataset_ctgan.csv', index=False)

print("Merged dataset saved as 'merged_dataset_ctgan.csv'")