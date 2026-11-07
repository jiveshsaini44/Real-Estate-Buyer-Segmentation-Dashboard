
from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ==========================================================
# STEP 4: DATA PREPARATION FOR BUYER CLUSTERING
# ==========================================================

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "outputs" / "buyer_integrated.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "clustering_preparation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------
# 1. LOAD INTEGRATED BUYER DATASET
# ----------------------------------------------------------

print("\n========== STEP 4: CLUSTERING DATA PREPARATION ==========")

df = pd.read_csv(INPUT_FILE)

print("\nOriginal dataset shape:", df.shape)
print("Unique buyers:", df["client_id"].nunique())


# ----------------------------------------------------------
# 2. VALIDATE REQUIRED COLUMNS
# ----------------------------------------------------------

required_columns = [
    "client_id",
    "client_type",
    "age",
    "acquisition_purpose",
    "loan_applied",
    "referral_channel",
    "satisfaction_score",
    "property_count",
    "average_property_price",
    "average_property_area",
    "property_categories",
    "country",
    "region",
    "gender"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

if df["client_id"].duplicated().any():
    raise ValueError(
        "Duplicate client IDs found. Each buyer must appear once."
    )


# ----------------------------------------------------------
# 3. SELECT FEATURES FOR CLUSTERING
# ----------------------------------------------------------

numeric_features = [
    "age",
    "satisfaction_score",
    "property_count",
    "average_property_price",
    "average_property_area",
    "property_categories"
]

categorical_features = [
    "client_type",
    "acquisition_purpose",
    "loan_applied",
    "referral_channel"
]

# Keep identifiers and geographic/demographic information
# separately for later cluster profiling.

profile_columns = [
    "client_id",
    "country",
    "region",
    "gender",
    "client_type",
    "acquisition_purpose"
]

profile_data = df[profile_columns].copy()

# Select only the clustering features.
clustering_data = df[
    numeric_features + categorical_features
].copy()


# ----------------------------------------------------------
# 4. DATA TYPE CONVERSION
# ----------------------------------------------------------

for col in numeric_features:
    clustering_data[col] = pd.to_numeric(
        clustering_data[col],
        errors="coerce"
    )

for col in categorical_features:
    clustering_data[col] = (
        clustering_data[col]
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
        .astype(object)
    )


# ----------------------------------------------------------
# 5. FEATURE ENGINEERING
# ----------------------------------------------------------

# Average property price relative to average property area.
# This is a property-level price intensity indicator.

clustering_data["price_per_sqft"] = (
    clustering_data["average_property_price"]
    / clustering_data["average_property_area"].replace(0, np.nan)
)

numeric_features.append("price_per_sqft")


# ----------------------------------------------------------
# 6. REMOVE INVALID NUMERIC VALUES
# ----------------------------------------------------------

clustering_data[numeric_features] = (
    clustering_data[numeric_features]
    .replace([np.inf, -np.inf], np.nan)
)

# Missing numeric values will be filled using median.
# Missing categorical values will be filled using mode.


# ----------------------------------------------------------
# 7. BUILD PREPROCESSING PIPELINE
# ----------------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ],
    remainder="drop"
)


# ----------------------------------------------------------
# 8. TRANSFORM DATA
# ----------------------------------------------------------

X_prepared = preprocessor.fit_transform(clustering_data)

# Obtain the transformed feature names.
feature_names = preprocessor.get_feature_names_out()

prepared_df = pd.DataFrame(
    X_prepared,
    columns=feature_names,
    index=df.index
)

# Keep buyer IDs separately for traceability.
prepared_df.insert(
    0,
    "client_id",
    df["client_id"].values
)


# ----------------------------------------------------------
# 9. SAVE OUTPUT FILES
# ----------------------------------------------------------

# A. Selected features before encoding/scaling.
clustering_data.insert(
    0,
    "client_id",
    df["client_id"].values
)

clustering_data.to_csv(
    OUTPUT_DIR / "clustering_input_raw.csv",
    index=False
)

# B. Final model-ready dataset.
prepared_df.to_csv(
    OUTPUT_DIR / "clustering_input_prepared.csv",
    index=False
)

# C. Geographic and demographic information for profiling.
profile_data.to_csv(
    OUTPUT_DIR / "cluster_profile_data.csv",
    index=False
)

# D. List of final transformed feature names.
pd.DataFrame({
    "feature_name": feature_names
}).to_csv(
    OUTPUT_DIR / "clustering_feature_names.csv",
    index=False
)

# E. Save the preprocessing pipeline for future use.
joblib.dump(
    preprocessor,
    OUTPUT_DIR / "clustering_preprocessor.joblib"
)


# ----------------------------------------------------------
# 10. VALIDATION AND SUMMARY
# ----------------------------------------------------------

print("\n========== PREPARATION SUMMARY ==========")

print("\nOriginal buyer records:", len(df))

print(
    "Unique buyer records in prepared dataset:",
    prepared_df["client_id"].nunique()
)

print("\nNumerical features:")
for feature in numeric_features:
    print("-", feature)

print("\nCategorical features:")
for feature in categorical_features:
    print("-", feature)

print("\nFinal transformed dataset shape:", X_prepared.shape)

print(
    "\nMissing values in prepared features:",
    prepared_df.drop(columns=["client_id"]).isna().sum().sum()
)

print(
    "Duplicate buyer IDs in prepared dataset:",
    prepared_df["client_id"].duplicated().sum()
)

print("\nSaved files:")
for file in sorted(OUTPUT_DIR.iterdir()):
    print("-", file.name)

print("\nSTEP 4 DATA PREPARATION COMPLETED SUCCESSFULLY!")
print("The dataset is now ready for clustering algorithms.")
