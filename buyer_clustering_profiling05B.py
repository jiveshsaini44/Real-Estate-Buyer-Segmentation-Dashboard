
from pathlib import Path
import pandas as pd
import numpy as np

# ==========================================================
# STEP 5B: BUYER CLUSTER PROFILING
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

# Change this path if your Step 5A output folder is different.
INPUT_DIR = BASE_DIR / "outputs" / "buyer_segmentation"

OUTPUT_DIR = BASE_DIR / "outputs" / "cluster_profiling"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\n========== STEP 5B: BUYER CLUSTER PROFILING ==========")

# ----------------------------------------------------------
# 1. LOAD SEGMENTED BUYER DATA
# ----------------------------------------------------------

file_path = INPUT_DIR / "kmeans_segmented_buyers.csv"

df = pd.read_csv(file_path)

print("\nDataset loaded successfully.")
print("Total buyers:", len(df))
print("Columns:", list(df.columns))

# ----------------------------------------------------------
# 2. IDENTIFY CLUSTER COLUMN
# ----------------------------------------------------------

possible_cluster_columns = [
    "Cluster",
    "cluster",
    "Cluster_Label",
    "cluster_label"
]

cluster_column = next(
    (col for col in possible_cluster_columns if col in df.columns),
    None
)

if cluster_column is None:
    raise ValueError(
        "Cluster column not found. Check the segmented buyer CSV."
    )

df[cluster_column] = pd.to_numeric(
    df[cluster_column],
    errors="raise"
).astype(int)

# ----------------------------------------------------------
# 3. NUMERICAL CLUSTER PROFILE
# ----------------------------------------------------------

numeric_columns = [
    "age",
    "satisfaction_score",
    "property_count",
    "total_property_value",
    "average_property_price",
    "average_property_area",
    "property_categories"
]

numeric_columns = [
    col for col in numeric_columns if col in df.columns
]

numeric_profile = (
    df.groupby(cluster_column)[numeric_columns]
    .agg(["mean", "median", "min", "max"])
    .round(2)
)

numeric_profile.to_csv(
    OUTPUT_DIR / "01_numeric_cluster_profile.csv"
)

# ----------------------------------------------------------
# 4. CATEGORICAL CLUSTER PROFILE
# ----------------------------------------------------------

categorical_columns = [
    "client_type",
    "acquisition_purpose",
    "loan_applied",
    "referral_channel"
]

categorical_columns = [
    col for col in categorical_columns if col in df.columns
]

for col in categorical_columns:

    profile = (
        pd.crosstab(
            df[cluster_column],
            df[col],
            normalize="index"
        ) * 100
    ).round(2)

    profile.to_csv(
        OUTPUT_DIR / f"02_{col}_profile.csv"
    )

# ----------------------------------------------------------
# 5. GEOGRAPHIC CLUSTER PROFILE
# ----------------------------------------------------------

for col in ["country", "region"]:

    if col in df.columns:

        profile = (
            pd.crosstab(
                df[cluster_column],
                df[col],
                normalize="index"
            ) * 100
        ).round(2)

        profile.to_csv(
            OUTPUT_DIR / f"03_{col}_profile.csv"
        )

# ----------------------------------------------------------
# 6. CLUSTER SIZE AND PERCENTAGE
# ----------------------------------------------------------

cluster_sizes = (
    df[cluster_column]
    .value_counts()
    .sort_index()
    .rename("Buyer_Count")
    .to_frame()
)

cluster_sizes["Percentage"] = (
    cluster_sizes["Buyer_Count"] / len(df) * 100
).round(2)

cluster_sizes.to_csv(
    OUTPUT_DIR / "04_cluster_sizes.csv"
)

# ----------------------------------------------------------
# 7. INVESTMENT BEHAVIOR SUMMARY
# ----------------------------------------------------------

investment_summary = df.groupby(cluster_column).agg(
    Buyer_Count=("client_id", "count"),
    Average_Age=("age", "mean"),
    Average_Property_Count=("property_count", "mean"),
    Average_Total_Property_Value=("total_property_value", "mean"),
    Average_Property_Price=("average_property_price", "mean"),
    Average_Satisfaction=("satisfaction_score", "mean")
).round(2)

investment_summary.to_csv(
    OUTPUT_DIR / "05_investment_behavior_summary.csv"
)

# ----------------------------------------------------------
# 8. TOP GEOGRAPHIC LOCATIONS
# ----------------------------------------------------------

for col in ["country", "region"]:

    if col in df.columns:

        top_locations = (
            df.groupby([cluster_column, col])
            .size()
            .reset_index(name="Buyer_Count")
            .sort_values(
                [cluster_column, "Buyer_Count"],
                ascending=[True, False]
            )
            .groupby(cluster_column)
            .head(5)
        )

        top_locations.to_csv(
            OUTPUT_DIR / f"06_top_{col}_by_cluster.csv",
            index=False
        )

# ----------------------------------------------------------
# 9. VALIDATION
# ----------------------------------------------------------

print("\n========== CLUSTER SIZE DISTRIBUTION ==========")
print(cluster_sizes)

print("\n========== INVESTMENT BEHAVIOR SUMMARY ==========")
print(investment_summary)

print("\n========== NUMERICAL CLUSTER PROFILE ==========")
print(numeric_profile)

print("\n========== CATEGORICAL CLUSTER PROFILES ==========")

for col in categorical_columns:
    print(f"\n{col.upper()}")
    print(pd.read_csv(
        OUTPUT_DIR / f"02_{col}_profile.csv",
        index_col=0
    ))

print("\n========== FINAL VALIDATION ==========")

print("Original buyers:", len(df))
print("Segmented buyers:", df[cluster_column].notna().sum())
print("Unique assigned buyers:", df["client_id"].nunique())
print("Number of clusters:", df[cluster_column].nunique())
print("Missing cluster labels:", df[cluster_column].isna().sum())
print("Duplicate buyer IDs:", df["client_id"].duplicated().sum())

print("\nGenerated profiling files:")

for file in sorted(OUTPUT_DIR.iterdir()):
    print("-", file.name)

print("\nSTEP 5B CLUSTER PROFILING COMPLETED SUCCESSFULLY!")
