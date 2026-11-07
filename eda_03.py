
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
EDA_DIR = OUTPUT_DIR / "eda"

EDA_DIR.mkdir(parents=True, exist_ok=True)

FILE_PATH = OUTPUT_DIR / "buyer_integrated.csv"

# ------------------------------------------
# 1. Load integrated dataset
# ------------------------------------------

df = pd.read_csv(FILE_PATH)

print("\n========== DATASET OVERVIEW ==========")

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst five records:")
print(df.head())

# ------------------------------------------
# 2. Basic statistical analysis
# ------------------------------------------

print("\n========== DATASET INFORMATION ==========")

print(df.info())

print("\nNumerical summary:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate client IDs:", df["client_id"].duplicated().sum())

# ------------------------------------------
# 3. Prepare analysis columns
# ------------------------------------------

categorical_columns = [
    "client_type",
    "gender",
    "country",
    "region",
    "acquisition_purpose",
    "loan_applied",
    "referral_channel"
]

for column in categorical_columns:
    df[column] = df[column].fillna("Unknown").astype(str).str.strip()

numeric_columns = [
    "age",
    "satisfaction_score",
    "property_count",
    "total_property_value",
    "average_property_price",
    "average_property_area"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Exclude impossible ages from age-based analysis
df.loc[
    ~df["age"].between(18, 100),
    "age"
] = np.nan

# ------------------------------------------
# 4. Buyer demographic analysis
# ------------------------------------------

print("\n========== BUYER DEMOGRAPHICS ==========")

print("\nBuyer type distribution:")
print(df["client_type"].value_counts())

print("\nGender distribution:")
print(df["gender"].value_counts())

print("\nAge summary:")
print(df["age"].describe())

print("\nBuyer type percentage:")
print(
    df["client_type"].value_counts(normalize=True) * 100
)

# ------------------------------------------
# 5. Investment behavior analysis
# ------------------------------------------

print("\n========== INVESTMENT BEHAVIOR ==========")

print("\nAcquisition purpose:")
print(df["acquisition_purpose"].value_counts())

print("\nAcquisition purpose percentage:")
print(
    df["acquisition_purpose"].value_counts(normalize=True) * 100
)

print("\nAverage property value by acquisition purpose:")

purpose_analysis = (
    df.groupby("acquisition_purpose")
    .agg(
        buyer_count=("client_id", "nunique"),
        average_property_value=("total_property_value", "mean"),
        average_property_price=("average_property_price", "mean"),
        average_property_count=("property_count", "mean")
    )
    .reset_index()
)

print(purpose_analysis)

purpose_analysis.to_csv(
    EDA_DIR / "investment_behavior.csv",
    index=False
)

# ------------------------------------------
# 6. Financing behavior analysis
# ------------------------------------------

print("\n========== FINANCING ANALYSIS ==========")

print("\nLoan application distribution:")
print(df["loan_applied"].value_counts())

loan_analysis = pd.crosstab(
    df["acquisition_purpose"],
    df["loan_applied"],
    normalize="index"
) * 100

print("\nLoan application by acquisition purpose (%):")
print(loan_analysis.round(2))

loan_analysis.to_csv(
    EDA_DIR / "loan_behavior.csv"
)

# ------------------------------------------
# 7. Geographic analysis
# ------------------------------------------

print("\n========== GEOGRAPHIC ANALYSIS ==========")

print("\nTop 10 countries:")
print(df["country"].value_counts().head(10))

print("\nTop 10 regions:")
print(df["region"].value_counts().head(10))

geographic_analysis = (
    df.groupby(["country", "region"])
    .agg(
        buyer_count=("client_id", "nunique"),
        investment_buyers=(
            "acquisition_purpose",
            lambda x: (x == "Investment").sum()
        ),
        average_property_value=("total_property_value", "mean")
    )
    .reset_index()
)

geographic_analysis.to_csv(
    EDA_DIR / "geographic_analysis.csv",
    index=False
)

# ------------------------------------------
# 8. Property investment analysis
# ------------------------------------------

print("\n========== PROPERTY ANALYSIS ==========")

print("\nProperty count summary:")
print(df["property_count"].describe())

print("\nTotal property value summary:")
print(df["total_property_value"].describe())

print("\nAverage property area summary:")
print(df["average_property_area"].describe())

property_analysis = (
    df.groupby("client_type")
    .agg(
        buyer_count=("client_id", "nunique"),
        average_properties=("property_count", "mean"),
        average_total_value=("total_property_value", "mean"),
        average_property_price=("average_property_price", "mean"),
        average_area=("average_property_area", "mean")
    )
    .reset_index()
)

print("\nProperty analysis by buyer type:")
print(property_analysis)

property_analysis.to_csv(
    EDA_DIR / "property_analysis.csv",
    index=False
)

# ------------------------------------------
# 9. Satisfaction analysis
# ------------------------------------------

print("\n========== SATISFACTION ANALYSIS ==========")

print(df["satisfaction_score"].describe())

satisfaction_analysis = (
    df.groupby("acquisition_purpose")
    .agg(
        average_satisfaction=("satisfaction_score", "mean"),
        buyer_count=("client_id", "nunique")
    )
    .reset_index()
)

print("\nSatisfaction by acquisition purpose:")
print(satisfaction_analysis)

satisfaction_analysis.to_csv(
    EDA_DIR / "satisfaction_analysis.csv",
    index=False
)

# ------------------------------------------
# 10. Visualization settings
# ------------------------------------------

sns.set_theme(style="whitegrid")

def save_chart(filename):
    plt.tight_layout()
    plt.savefig(
        EDA_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

# ------------------------------------------
# CHART 1: Buyer type distribution
# ------------------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="client_type",
    order=df["client_type"].value_counts().index
)

plt.title("Buyer Distribution by Client Type")
plt.xlabel("Client Type")
plt.ylabel("Number of Buyers")

save_chart("01_buyer_type.png")

# ------------------------------------------
# CHART 2: Acquisition purpose
# ------------------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="acquisition_purpose",
    order=df["acquisition_purpose"].value_counts().index
)

plt.title("Buyer Distribution by Acquisition Purpose")
plt.xlabel("Acquisition Purpose")
plt.ylabel("Number of Buyers")

save_chart("02_acquisition_purpose.png")

# ------------------------------------------
# CHART 3: Age distribution
# ------------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="age",
    bins=20,
    kde=True
)

plt.title("Buyer Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Buyers")

save_chart("03_age_distribution.png")

# ------------------------------------------
# CHART 4: Gender distribution
# ------------------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="gender",
    order=df["gender"].value_counts().index
)

plt.title("Buyer Distribution by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Buyers")

save_chart("04_gender_distribution.png")

# ------------------------------------------
# CHART 5: Loan application behavior
# ------------------------------------------

plt.figure(figsize=(9, 5))

sns.countplot(
    data=df,
    x="acquisition_purpose",
    hue="loan_applied"
)

plt.title("Loan Application by Acquisition Purpose")
plt.xlabel("Acquisition Purpose")
plt.ylabel("Number of Buyers")

save_chart("05_loan_behavior.png")

# ------------------------------------------
# CHART 6: Top 10 countries
# ------------------------------------------

top_countries = df["country"].value_counts().head(10)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=top_countries.values,
    y=top_countries.index
)

plt.title("Top 10 Countries by Buyer Count")
plt.xlabel("Number of Buyers")
plt.ylabel("Country")

save_chart("06_top_countries.png")

# ------------------------------------------
# CHART 7: Top 10 regions
# ------------------------------------------

top_regions = df["region"].value_counts().head(10)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=top_regions.values,
    y=top_regions.index
)

plt.title("Top 10 Regions by Buyer Count")
plt.xlabel("Number of Buyers")
plt.ylabel("Region")

save_chart("07_top_regions.png")

# ------------------------------------------
# CHART 8: Property value by buyer purpose
# ------------------------------------------

plt.figure(figsize=(9, 5))

sns.boxplot(
    data=df,
    x="acquisition_purpose",
    y="total_property_value"
)

plt.title("Total Property Value by Acquisition Purpose")
plt.xlabel("Acquisition Purpose")
plt.ylabel("Total Property Value")

save_chart("08_property_value_by_purpose.png")

# ------------------------------------------
# CHART 9: Average property count by client type
# ------------------------------------------

plt.figure(figsize=(8, 5))

sns.barplot(
    data=df,
    x="client_type",
    y="property_count",
    errorbar=None
)

plt.title("Average Properties per Buyer by Client Type")
plt.xlabel("Client Type")
plt.ylabel("Average Property Count")

save_chart("09_property_count_by_type.png")

# ------------------------------------------
# CHART 10: Satisfaction distribution
# ------------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="satisfaction_score",
    bins=10,
    kde=True
)

plt.title("Buyer Satisfaction Distribution")
plt.xlabel("Satisfaction Score")
plt.ylabel("Number of Buyers")

save_chart("10_satisfaction_distribution.png")

# ------------------------------------------
# 11. Save final EDA dataset
# ------------------------------------------

df.to_csv(
    EDA_DIR / "eda_analysis_dataset.csv",
    index=False
)

# ------------------------------------------
# 12. Final summary
# ------------------------------------------

print("\n========== EDA COMPLETED ==========")

print("Total buyers analyzed:", len(df))

print("Total charts generated: 10")

print("EDA output directory:", EDA_DIR)

print("\nGenerated files:")
for file in sorted(EDA_DIR.iterdir()):
    print(file.name)

print("\nStep 3 EDA analysis completed successfully!")
