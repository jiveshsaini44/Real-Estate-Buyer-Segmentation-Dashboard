
from pathlib import Path
import pandas as pd
import numpy as np

# ==========================================================
# STEP 5C: BUYER SEGMENT INTERPRETATION
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_BASE = BASE_DIR / "outputs"

# Locate the segmented buyer dataset.
matches = list(
    OUTPUT_BASE.rglob("kmeans_segmented_buyers.csv")
)

if not matches:
    raise FileNotFoundError(
        "Could not find kmeans_segmented_buyers.csv "
        "inside the outputs folder."
    )

INPUT_FILE = matches[0]

OUTPUT_DIR = OUTPUT_BASE / "segment_interpretation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\n========== STEP 5C: BUYER SEGMENT INTERPRETATION ==========")

# ----------------------------------------------------------
# 1. LOAD SEGMENTED BUYER DATA
# ----------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("\nDataset loaded:", INPUT_FILE)
print("Total buyers:", len(df))

cluster_candidates = [
    "Cluster", "cluster", "Cluster_Label", "cluster_label"
]

cluster_col = next(
    (c for c in cluster_candidates if c in df.columns),
    None
)

if cluster_col is None:
    raise ValueError("Cluster column not found.")

if "client_id" not in df.columns:
    raise ValueError("client_id column not found.")

if df["client_id"].duplicated().any():
    raise ValueError("Duplicate buyer IDs found.")

if df[cluster_col].isna().any():
    raise ValueError("Some buyers have no cluster assignment.")

# ----------------------------------------------------------
# 2. DEFINE NUMERICAL AND CATEGORICAL FEATURES
# ----------------------------------------------------------

numeric_features = [
    "age",
    "satisfaction_score",
    "property_count",
    "total_property_value",
    "average_property_price",
    "average_property_area",
    "property_categories"
]

categorical_features = [
    "client_type",
    "acquisition_purpose",
    "loan_applied",
    "referral_channel",
    "country",
    "region"
]

numeric_features = [
    c for c in numeric_features if c in df.columns
]

categorical_features = [
    c for c in categorical_features if c in df.columns
]

for col in numeric_features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ----------------------------------------------------------
# 3. CALCULATE CLUSTER SIZE
# ----------------------------------------------------------

cluster_sizes = (
    df.groupby(cluster_col)
    .size()
    .reset_index(name="Buyer_Count")
)

cluster_sizes["Percentage"] = (
    cluster_sizes["Buyer_Count"] / len(df) * 100
).round(2)

# ----------------------------------------------------------
# 4. CALCULATE NUMERICAL CHARACTERISTICS
# ----------------------------------------------------------

numeric_profile = (
    df.groupby(cluster_col)[numeric_features]
    .agg(["mean", "median"])
    .round(2)
)

numeric_profile.to_csv(
    OUTPUT_DIR / "01_segment_numeric_characteristics.csv"
)

# Create an easy-to-read average summary.
mean_profile = (
    df.groupby(cluster_col)[numeric_features]
    .mean()
    .round(2)
    .reset_index()
)

# ----------------------------------------------------------
# 5. ASSIGN DESCRIPTIVE SEGMENT NAMES
# ----------------------------------------------------------

# Names are based on average property price.
# These are descriptive labels, not income classifications.

if "average_property_price" in df.columns:

    price_order = (
        df.groupby(cluster_col)["average_property_price"]
        .mean()
        .sort_values()
    )

    cluster_ids = price_order.index.tolist()

    if len(cluster_ids) == 2:
        segment_names = {
            cluster_ids[0]:
                "Lower-Average-Price, Multiple-Property Buyers",
            cluster_ids[1]:
                "Higher-Average-Price, Larger-Property Buyers"
        }
    else:
        segment_names = {
            cluster_id:
                f"Property Price Segment {rank + 1}"
            for rank, cluster_id in enumerate(cluster_ids)
        }

else:
    segment_names = {
        cluster_id: f"Buyer Segment {cluster_id}"
        for cluster_id in df[cluster_col].unique()
    }

# ----------------------------------------------------------
# 6. CATEGORICAL BEHAVIOR PROFILES
# ----------------------------------------------------------

categorical_profiles = {}

for col in categorical_features:

    profile = (
        pd.crosstab(
            df[cluster_col],
            df[col],
            normalize="index"
        ) * 100
    ).round(2)

    categorical_profiles[col] = profile

    profile.to_csv(
        OUTPUT_DIR / f"02_{col}_segment_profile.csv"
    )

# ----------------------------------------------------------
# 7. CREATE SEGMENT CHARACTERISTICS SUMMARY
# ----------------------------------------------------------

summary_rows = []

for cluster_id in sorted(df[cluster_col].unique()):

    group = df[df[cluster_col] == cluster_id]

    row = {
        "Cluster": cluster_id,
        "Segment_Name": segment_names[cluster_id],
        "Buyer_Count": len(group),
        "Percentage": round(
            len(group) / len(df) * 100, 2
        )
    }

    for col in numeric_features:
        row[f"Average_{col}"] = round(
            group[col].mean(), 2
        )

    for col in categorical_features:
        if group[col].notna().any():
            row[f"Most_Common_{col}"] = (
                group[col].mode().iloc[0]
            )

    summary_rows.append(row)

segment_summary = pd.DataFrame(summary_rows)

segment_summary.to_csv(
    OUTPUT_DIR / "03_buyer_segment_summary.csv",
    index=False
)

# ----------------------------------------------------------
# 8. GENERATE BUSINESS INSIGHTS AND RECOMMENDATIONS
# ----------------------------------------------------------

recommendation_rows = []

for cluster_id in sorted(df[cluster_col].unique()):

    group = df[df[cluster_col] == cluster_id]

    avg_price = group["average_property_price"].mean()
    avg_area = group["average_property_area"].mean()
    avg_count = group["property_count"].mean()

    name = segment_names[cluster_id]

    if "average_property_price" in df.columns:
        price_description = (
            "lower-average-price"
            if avg_price <= df["average_property_price"].median()
            else "higher-average-price"
        )
    else:
        price_description = "unclassified-price"

    if "average_property_area" in df.columns:
        area_description = (
            "smaller-area"
            if avg_area <= df["average_property_area"].median()
            else "larger-area"
        )
    else:
        area_description = "unclassified-area"

    if "property_count" in df.columns:
        count_description = (
            "higher property count"
            if avg_count >= df["property_count"].median()
            else "lower property count"
        )
    else:
        count_description = "unclassified property count"

    insight = (
        f"This segment contains {len(group)} buyers. "
        f"It shows {price_description} purchasing behavior, "
        f"{area_description} preferences and {count_description}."
    )

    recommendation = (
        "Explore property offerings and marketing messages "
        "that reflect this segment's observed purchasing "
        "characteristics. Validate demand before changing "
        "product or investment strategy."
    )

    if "average_property_price" in df.columns:
        if avg_price > df["average_property_price"].median():
            recommendation = (
                "Prioritize analysis of larger and higher-priced "
                "property offerings for this segment. Assess "
                "customer demand and financing needs before "
                "designing targeted campaigns."
            )
        else:
            recommendation = (
                "Explore a broader range of comparatively "
                "lower-priced properties and portfolio options. "
                "Use property count and buyer preferences to "
                "design relevant offerings."
            )

    recommendation_rows.append({
        "Cluster": cluster_id,
        "Segment_Name": name,
        "Key_Insight": insight,
        "Business_Recommendation": recommendation
    })

recommendations = pd.DataFrame(recommendation_rows)

recommendations.to_csv(
    OUTPUT_DIR / "04_segment_business_recommendations.csv",
    index=False
)

# ----------------------------------------------------------
# 9. GENERATE WRITTEN INTERPRETATION REPORT
# ----------------------------------------------------------

report_lines = []

report_lines.append("BUYER SEGMENTATION INTERPRETATION REPORT")
report_lines.append("=" * 55)

report_lines.append(f"\nTotal buyers analyzed: {len(df)}")
report_lines.append(
    f"Number of clusters: {df[cluster_col].nunique()}"
)

for cluster_id in sorted(df[cluster_col].unique()):

    group = df[df[cluster_col] == cluster_id]

    name = segment_names[cluster_id]

    report_lines.append("\n" + "=" * 55)
    report_lines.append(f"CLUSTER {cluster_id}: {name}")
    report_lines.append("=" * 55)

    report_lines.append(
        f"Buyer count: {len(group)}"
    )

    report_lines.append(
        f"Buyer percentage: {len(group) / len(df) * 100:.2f}%"
    )

    report_lines.append("\nNumerical characteristics:")

    for col in numeric_features:
        report_lines.append(
            f"- Average {col}: {group[col].mean():,.2f}"
        )

    report_lines.append("\nMost common buyer characteristics:")

    for col in categorical_features:
        if group[col].notna().any():
            mode = group[col].mode().iloc[0]
            percentage = (
                group[col].value_counts(normalize=True).iloc[0] * 100
            )

            report_lines.append(
                f"- {col}: {mode} ({percentage:.2f}%)"
            )

    recommendation = recommendations.loc[
        recommendations["Cluster"] == cluster_id,
        "Business_Recommendation"
    ].iloc[0]

    report_lines.append("\nBusiness recommendation:")
    report_lines.append(recommendation)

# Add methodological caution.
report_lines.extend([
    "\n" + "=" * 55,
    "INTERPRETATION LIMITATIONS",
    "=" * 55,
    "- Segment names describe observed purchasing patterns.",
    "- Property value is not a direct measure of buyer income or wealth.",
    "- Similar cluster averages do not prove identical buyer behavior.",
    "- K-Means currently has a low Silhouette Score.",
    "- Segment labels remain provisional until model comparison."
])

report_path = OUTPUT_DIR / "05_segment_interpretation_report.txt"

with open(report_path, "w", encoding="utf-8") as file:
    file.write("\n".join(report_lines))

# ----------------------------------------------------------
# 10. FINAL VALIDATION
# ----------------------------------------------------------

print("\n========== SEGMENT INTERPRETATION SUMMARY ==========")

print(
    segment_summary[
        ["Cluster", "Segment_Name", "Buyer_Count", "Percentage"]
    ].to_string(index=False)
)

print("\n========== BUSINESS RECOMMENDATIONS ==========")

print(recommendations.to_string(index=False))

print("\n========== FINAL VALIDATION ==========")

print("Original buyers:", len(df))
print("Segmented buyers:", df[cluster_col].notna().sum())
print("Unique assigned buyers:", df[cluster_col].nunique())
print("Missing cluster assignments:", df[cluster_col].isna().sum())
print("Duplicate buyer IDs:", df["client_id"].duplicated().sum())

print("\nGenerated files:")

for file in sorted(OUTPUT_DIR.iterdir()):
    print("-", file.name)

print("\nSTEP 5C SEGMENT INTERPRETATION COMPLETED SUCCESSFULLY!")
