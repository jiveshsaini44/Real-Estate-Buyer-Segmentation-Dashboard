
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from scipy.cluster.hierarchy import linkage, fcluster

from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

warnings.filterwarnings("ignore")

# --------------------------------------------------
# STEP 5E: FINAL BUYER SEGMENTATION VALIDATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

PREPARED_FILE = (
    BASE_DIR / "outputs" / "clustering_preparation"
    / "clustering_input_prepared.csv"
)
PROFILE_FILE = (
    BASE_DIR / "outputs"
    / "buyer_integrated.csv"
)


KMEANS_ASSIGNMENT_FILE = (
    BASE_DIR / "outputs" / "buyer_segmentation"
    / "kmeans_buyer_assignments.csv"
)

HIERARCHICAL_ASSIGNMENT_FILE = (
    BASE_DIR / "outputs" / "hierarchical_clustering"
    / "hierarchical_buyer_assignments.csv"
)

OUTPUT_DIR = BASE_DIR / "outputs" / "final_segmentation_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# HELPER: FIND THE CLUSTER LABEL COLUMN
# --------------------------------------------------

def find_cluster_column(df, algorithm_name):

    possible_columns = [
        col for col in df.columns
        if "cluster" in col.lower()
        and col.lower() not in ["cluster_size"]
    ]

    # Prefer a cluster column with exactly two groups
    # for the existing K-Means K=2 assignment file.
    if algorithm_name == "K-Means":
        two_group_columns = [
            col for col in possible_columns
            if df[col].nunique(dropna=True) == 2
        ]

        if two_group_columns:
            return two_group_columns[0]

    if possible_columns:
        return possible_columns[0]

    raise ValueError(
        f"Could not find a cluster-label column in {algorithm_name} file. "
        f"Available columns: {list(df.columns)}"
    )


# --------------------------------------------------
# 1. LOAD PREPARED DATA
# --------------------------------------------------

print("\n========== STEP 5E: FINAL SEGMENTATION VALIDATION ==========")

required_files = [
    PREPARED_FILE,
    PROFILE_FILE,
    KMEANS_ASSIGNMENT_FILE,
    HIERARCHICAL_ASSIGNMENT_FILE
]

for file_path in required_files:
    if not file_path.exists():
        raise FileNotFoundError(f"Required file not found: {file_path}")

prepared_df = pd.read_csv(PREPARED_FILE)
profile_df = pd.read_csv(PROFILE_FILE)

buyer_ids = prepared_df["client_id"]

X = prepared_df.drop(columns=["client_id"])
X = X.apply(pd.to_numeric, errors="coerce")

if X.isnull().any().any():
    raise ValueError("Missing values found in prepared clustering features.")

X_values = X.to_numpy(dtype=float)

print("\n========== DATA SUMMARY ==========")
print("Total buyers:", len(prepared_df))
print("Clustering features:", X_values.shape[1])
print("Unique buyer IDs:", buyer_ids.nunique())


# --------------------------------------------------
# 2. LOAD EXISTING K-MEANS ASSIGNMENTS
# --------------------------------------------------

kmeans_df = pd.read_csv(KMEANS_ASSIGNMENT_FILE)

if "client_id" not in kmeans_df.columns:
    raise ValueError("client_id missing from K-Means assignments.")

kmeans_cluster_col = find_cluster_column(kmeans_df, "K-Means")

kmeans_assignments = kmeans_df[
    ["client_id", kmeans_cluster_col]
].copy()

kmeans_assignments = kmeans_assignments.rename(
    columns={kmeans_cluster_col: "kmeans_cluster"}
)

if kmeans_assignments["client_id"].duplicated().any():
    raise ValueError("Duplicate buyer IDs in K-Means assignments.")

if kmeans_assignments["kmeans_cluster"].isna().any():
    raise ValueError("Missing K-Means cluster labels.")

print("\nK-Means assignments loaded.")
print("K-Means cluster count:",
      kmeans_assignments["kmeans_cluster"].nunique())


# --------------------------------------------------
# 3. GENERATE HIERARCHICAL K=2 FOR FAIR COMPARISON
# --------------------------------------------------

print("\n========== GENERATING HIERARCHICAL K=2 ==========")

Z = linkage(
    X_values,
    method="ward",
    metric="euclidean"
)

hierarchical_k2_labels = fcluster(
    Z,
    t=2,
    criterion="maxclust"
)

hierarchical_k2_df = pd.DataFrame({
    "client_id": buyer_ids,
    "hierarchical_cluster_k2": hierarchical_k2_labels
})

if hierarchical_k2_df["hierarchical_cluster_k2"].nunique() != 2:
    raise ValueError("Hierarchical K=2 did not produce exactly two clusters.")

# Align both algorithms by buyer ID.
comparison_df = kmeans_assignments.merge(
    hierarchical_k2_df,
    on="client_id",
    how="inner",
    validate="one_to_one"
)

if len(comparison_df) != len(prepared_df):
    raise ValueError("Buyer mismatch between the two algorithms.")

# --------------------------------------------------
# 4. CALCULATE ALGORITHM AGREEMENT
# --------------------------------------------------

ari = adjusted_rand_score(
    comparison_df["kmeans_cluster"],
    comparison_df["hierarchical_cluster_k2"]
)

nmi = normalized_mutual_info_score(
    comparison_df["kmeans_cluster"],
    comparison_df["hierarchical_cluster_k2"]
)

print("\n========== ALGORITHM AGREEMENT ==========")
print("Adjusted Rand Index (ARI):", round(ari, 4))
print("Normalized Mutual Information (NMI):", round(nmi, 4))

agreement_df = pd.DataFrame([{
    "Compared_Buyers": len(comparison_df),
    "KMeans_Clusters": comparison_df["kmeans_cluster"].nunique(),
    "Hierarchical_Clusters": comparison_df[
        "hierarchical_cluster_k2"
    ].nunique(),
    "Adjusted_Rand_Index": ari,
    "Normalized_Mutual_Information": nmi
}])

agreement_df.to_csv(
    OUTPUT_DIR / "01_algorithm_agreement.csv",
    index=False
)

comparison_df.to_csv(
    OUTPUT_DIR / "02_kmeans_vs_hierarchical_k2_assignments.csv",
    index=False
)


# --------------------------------------------------
# 5. LOAD EXISTING HIERARCHICAL K=4 ASSIGNMENTS
# --------------------------------------------------

hierarchical_k4_df = pd.read_csv(HIERARCHICAL_ASSIGNMENT_FILE)

if "client_id" not in hierarchical_k4_df.columns:
    raise ValueError("client_id missing from hierarchical assignments.")

hierarchical_cluster_col = find_cluster_column(
    hierarchical_k4_df,
    "Hierarchical"
)

hierarchical_k4_df = hierarchical_k4_df[
    ["client_id", hierarchical_cluster_col]
].copy()

hierarchical_k4_df = hierarchical_k4_df.rename(
    columns={hierarchical_cluster_col: "hierarchical_cluster_k4"}
)

if hierarchical_k4_df["client_id"].duplicated().any():
    raise ValueError("Duplicate buyer IDs in hierarchical assignments.")

if hierarchical_k4_df["hierarchical_cluster_k4"].isna().any():
    raise ValueError("Missing hierarchical cluster labels.")

if len(hierarchical_k4_df) != len(prepared_df):
    raise ValueError("Hierarchical K=4 buyer count does not match input.")

if hierarchical_k4_df["hierarchical_cluster_k4"].nunique() != 4:
    raise ValueError("Expected four hierarchical clusters.")

print("\nHierarchical K=4 assignments loaded.")


# --------------------------------------------------
# 6. COMPARE CLUSTER EVALUATION METRICS
# --------------------------------------------------

print("\n========== COMPARING CLUSTER QUALITY ==========")

# K-Means K=2: use the actual saved assignments.
kmeans_labels = comparison_df["kmeans_cluster"].to_numpy()

# Hierarchical K=2: generated from the same feature matrix.
hierarchical_k2 = hierarchical_k2_labels

# Hierarchical K=4: existing selected solution.
hierarchical_k4 = (
    hierarchical_k4_df
    .set_index("client_id")
    .loc[buyer_ids, "hierarchical_cluster_k4"]
    .to_numpy()
)

evaluation_rows = []

solutions = [
    ("K-Means", 2, kmeans_labels),
    ("Hierarchical", 2, hierarchical_k2),
    ("Hierarchical", 4, hierarchical_k4)
]

for algorithm, k, labels in solutions:

    evaluation_rows.append({
        "Algorithm": algorithm,
        "K": k,
        "Actual_Clusters": len(np.unique(labels)),
        "Silhouette_Score": silhouette_score(X_values, labels),
        "Davies_Bouldin_Index": davies_bouldin_score(
            X_values, labels
        ),
        "Calinski_Harabasz_Score": calinski_harabasz_score(
            X_values, labels
        )
    })

evaluation_df = pd.DataFrame(evaluation_rows)

evaluation_df = evaluation_df.round(4)

evaluation_df.to_csv(
    OUTPUT_DIR / "03_final_algorithm_comparison.csv",
    index=False
)

print(evaluation_df.to_string(index=False))


# --------------------------------------------------
# 7. VALIDATE CLUSTER SIZES
# --------------------------------------------------

print("\n========== CLUSTER SIZE VALIDATION ==========")

kmeans_sizes = (
    comparison_df["kmeans_cluster"]
    .value_counts()
    .sort_index()
)

hierarchical_k2_sizes = (
    hierarchical_k2_df["hierarchical_cluster_k2"]
    .value_counts()
    .sort_index()
)

hierarchical_k4_sizes = (
    hierarchical_k4_df["hierarchical_cluster_k4"]
    .value_counts()
    .sort_index()
)

size_rows = []

for cluster, count in kmeans_sizes.items():
    size_rows.append({
        "Algorithm": "K-Means K=2",
        "Cluster": cluster,
        "Buyer_Count": count,
        "Buyer_Percentage": round(count / len(prepared_df) * 100, 2)
    })

for cluster, count in hierarchical_k2_sizes.items():
    size_rows.append({
        "Algorithm": "Hierarchical K=2",
        "Cluster": cluster,
        "Buyer_Count": count,
        "Buyer_Percentage": round(count / len(prepared_df) * 100, 2)
    })

for cluster, count in hierarchical_k4_sizes.items():
    size_rows.append({
        "Algorithm": "Hierarchical K=4",
        "Cluster": cluster,
        "Buyer_Count": count,
        "Buyer_Percentage": round(count / len(prepared_df) * 100, 2)
    })

size_df = pd.DataFrame(size_rows)

size_df.to_csv(
    OUTPUT_DIR / "04_cluster_size_comparison.csv",
    index=False
)

print(size_df.to_string(index=False))


# --------------------------------------------------
# 8. PROFILE HIERARCHICAL K=4 CLUSTERS
# --------------------------------------------------

print("\n========== HIERARCHICAL K=4 BUYER PROFILES ==========")

if "client_id" not in profile_df.columns:
    raise ValueError("client_id missing from cluster profile data.")

if profile_df["client_id"].duplicated().any():
    raise ValueError("Duplicate buyer IDs in cluster profile data.")

profile_with_clusters = profile_df.merge(
    hierarchical_k4_df,
    on="client_id",
    how="inner",
    validate="one_to_one"
)

if len(profile_with_clusters) != len(prepared_df):
    raise ValueError("Profile data does not cover all buyers.")

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
    col for col in numeric_columns
    if col in profile_with_clusters.columns
]

numeric_profile = (
    profile_with_clusters
    .groupby("hierarchical_cluster_k4")[numeric_columns]
    .mean()
    .round(2)
    .reset_index()
)

numeric_profile.to_csv(
    OUTPUT_DIR / "05_hierarchical_k4_numeric_profiles.csv",
    index=False
)

print("\nNumeric profile:")
print(numeric_profile.to_string(index=False))
# --------------------------------------------------
# 9. CATEGORICAL PROFILE
# --------------------------------------------------

categorical_candidates = [
    "client_type",
    "acquisition_purpose",
    "loan_applied",
    "referral_channel",
    "country",
    "region"
]

categorical_columns = [
    col for col in categorical_candidates
    if col in profile_with_clusters.columns
]

categorical_rows = []

for col in categorical_columns:

    for cluster in sorted(
        profile_with_clusters["hierarchical_cluster_k4"].unique()
    ):

        cluster_data = profile_with_clusters[
            profile_with_clusters["hierarchical_cluster_k4"] == cluster
        ]

        percentages = (
            cluster_data[col]
            .fillna("Unknown")
            .astype(str)
            .value_counts(normalize=True) * 100
        )

        for category, percentage in percentages.items():

            categorical_rows.append({
                "Cluster": cluster,
                "Feature": col,
                "Category": category,
                "Percentage": round(percentage, 2)
            })

categorical_profile = pd.DataFrame(categorical_rows)

categorical_profile.to_csv(
    OUTPUT_DIR / "06_hierarchical_k4_categorical_profiles.csv",
    index=False
)

print("\nCategorical profile saved.")


# --------------------------------------------------
# 10. INVESTIGATE THE SMALL CLUSTER
# --------------------------------------------------

print("\n========== SMALL CLUSTER INVESTIGATION ==========")

cluster_counts = (
    hierarchical_k4_df["hierarchical_cluster_k4"]
    .value_counts()
    .sort_values()
)

smallest_cluster = cluster_counts.index[0]
smallest_count = int(cluster_counts.iloc[0])
smallest_percentage = smallest_count / len(prepared_df) * 100

small_cluster_summary = numeric_profile[
    numeric_profile["hierarchical_cluster_k4"] == smallest_cluster
].copy()

small_cluster_summary.to_csv(
    OUTPUT_DIR / "07_smallest_cluster_profile.csv",
    index=False
)

print("Smallest cluster:", smallest_cluster)
print("Buyer count:", smallest_count)
print("Buyer percentage:", round(smallest_percentage, 2))

print("\nSmallest cluster numeric characteristics:")
print(small_cluster_summary.to_string(index=False))


# --------------------------------------------------
# 11. FINAL DATA VALIDATION
# --------------------------------------------------

print("\n========== FINAL VALIDATION ==========")

print("Original buyers:", len(prepared_df))
print("Unique original buyer IDs:", buyer_ids.nunique())

print(
    "K-Means K=2 assigned buyers:",
    comparison_df["client_id"].nunique()
)

print(
    "Hierarchical K=2 assigned buyers:",
    hierarchical_k2_df["client_id"].nunique()
)

print(
    "Hierarchical K=4 assigned buyers:",
    hierarchical_k4_df["client_id"].nunique()
)

print(
    "Missing K-Means assignments:",
    comparison_df["kmeans_cluster"].isna().sum()
)

print(
    "Missing Hierarchical K=4 assignments:",
    hierarchical_k4_df["hierarchical_cluster_k4"].isna().sum()
)

print(
    "Duplicate K-Means buyer IDs:",
    comparison_df["client_id"].duplicated().sum()
)

print(
    "Duplicate Hierarchical buyer IDs:",
    hierarchical_k4_df["client_id"].duplicated().sum()
)

assert len(comparison_df) == len(prepared_df)
assert len(hierarchical_k4_df) == len(prepared_df)
assert comparison_df["client_id"].nunique() == len(prepared_df)
assert hierarchical_k4_df["client_id"].nunique() == len(prepared_df)

# --------------------------------------------------
# 12. SAVE FINAL SUMMARY REPORT
# --------------------------------------------------

best_quality_row = evaluation_df.loc[
    evaluation_df["Silhouette_Score"].idxmax()
]

report_lines = [
    "FINAL BUYER SEGMENTATION VALIDATION REPORT",
    "=" * 55,
    "",
    f"Total buyers: {len(prepared_df)}",
    f"Clustering features: {X_values.shape[1]}",
    "",
    "ALGORITHM AGREEMENT",
    "-" * 30,
    f"Adjusted Rand Index: {ari:.4f}",
    f"Normalized Mutual Information: {nmi:.4f}",
    "",
    "EVALUATION RESULTS",
    "-" * 30,
    evaluation_df.to_string(index=False),
    "",
    "BEST SILHOUETTE RESULT",
    "-" * 30,
    f"Algorithm: {best_quality_row['Algorithm']}",
    f"K: {int(best_quality_row['K'])}",
    f"Silhouette: {best_quality_row['Silhouette_Score']:.4f}",
    "",
    "HIERARCHICAL K=4 CLUSTER SIZES",
    "-" * 30,
    size_df[
        size_df["Algorithm"] == "Hierarchical K=4"
    ].to_string(index=False),
    "",
    "SMALLEST HIERARCHICAL CLUSTER",
    "-" * 30,
    f"Cluster: {smallest_cluster}",
    f"Buyers: {smallest_count}",
    f"Percentage: {smallest_percentage:.2f}%",
    "",
    "INTERPRETATION NOTE",
    "-" * 30,
    "ARI and NMI measure agreement between algorithms,",
    "not commercial usefulness or segment quality.",
    "Cluster labels are arbitrary identifiers.",
    "Business interpretations must be based on buyer profiles.",
    "Low silhouette values indicate weak cluster separation.",
]

report_path = OUTPUT_DIR / "08_final_segmentation_validation_report.txt"

report_path.write_text(
    "\n".join(report_lines),
    encoding="utf-8"
)

print("\n========== STEP 5E COMPLETED ==========")
print("Output folder:", OUTPUT_DIR)
print("Final report:", report_path)
