
import pandas as pd
from pathlib import Path

file = Path("outputs/clustering_preparation/cluster_profile_data.csv")

df = pd.read_csv(file)

print("\n========== PROFILE DATA CHECK ==========")

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nColumn data types:")
print(df.dtypes)

print("\nFirst five rows:")
print(df.head().to_string())
