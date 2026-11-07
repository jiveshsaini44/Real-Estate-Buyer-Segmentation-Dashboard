
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

warnings.filterwarnings("ignore")

# --------------------------------------------------
# STEP 5D: HIERARCHICAL BUYER SEGMENTATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE_DIR
    / "outputs"
    / "clustering_preparation"
    / "clustering_input_prepared.csv"
)

PROFILE_FILE = (
    BASE_DIR
    / "outputs"
    / "clustering_preparation"
    / "cluster_profile_data.csv"
)

OUTPUT_DIR = BASE_DIR / "outputs" / "hierarchical_clustering"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Test cluster counts from 2 to 10
K_VALUES = range(2, 11)

print("\n========== STEP 5D: HIERARCHICAL CLUSTERING ==========")

# --------------------------------------------------
# 1. LOAD PREPARED DATA
# --------------------------------------------------

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

prepared_df = pd.read_csv(INPUT_FILE)

if "client_id" not in prepared_df.columns:
    raise ValueError("client_id column is missing from prepared dataset.")

buyer_ids = prepared_df["client_id"]

# Exclude buyer ID from clustering.
X = prepared_df.drop(columns=["client_id"])

# Ensure all clustering features are numeric.
X = X.apply(pd.to_numeric, errors="coerce")

if X.isnull().any().any():
    raise ValueError("Missing or non-numeric values found in clustering features.")

X_values = X.to_numpy(dtype=float)

print("\n========== INPUT SUMMARY ==========")
print("Total buyers:", len(X_values))
print("Clustering features:", X_values.shape[1])
print("Unique buyer IDs:", buyer_ids.nunique())

if buyer_ids.nunique() != len(buyer_ids):
    raise ValueError("Duplicate buyer IDs found.")

if len(X_values) < max(K_VALUES):
    raise ValueError("Not enough buyers for the selected cluster counts.")

# --------------------------------------------------
# 2. BUILD HIERARCHICAL TREE
# --------------------------------------------------

print("\n========== BUILDING HIERARCHICAL TREE ==========")

# Ward linkage merges clusters while minimizing
# the increase in within-cluster variance.
Z = linkage(
    X_values,
    method="ward",
    metric="euclidean"
)

print("Hierarchical tree successfully created.")

# --------------------------------------------------
# 3. EVALUATE DIFFERENT CLUSTER COUNTS
# --------------------------------------------------

print("\n========== CLUSTER EVALUATION ==========")

evaluation_results = []
cluster_assignments = {}

for k in K_VALUES:

    # Extract k clusters from the hierarchical tree.
    labels = fcluster(
        Z,
        t=k,
        criterion="maxclust"
    )

    actual_clusters = np.unique(labels).size

    if actual_clusters < 2:
        print(f"K={k}: skipped because fewer than 2 clusters were formed.")
        continue

    silhouette = silhouette_score(X_values, labels)
    dbi = davies_bouldin_score(X_values, labels)
    ch = calinski_harabasz_score(X_values, labels)

    evaluation_results.append({
        "Requested_K": k,
        "Actual_Clusters": actual_clusters,
        "Silhouette_Score": silhouette,
        "Davies_Bouldin_Index": dbi,
        "Calinski_Harabasz_Score": ch
    })

    cluster_assignments[k] = labels

    print(
        f"K={k} | "
        f"Clusters={actual_clusters} | "
        f"Silhouette={silhouette:.4f} | "
        f"DBI={dbi:.4f} | "
        f"CH={ch:.2f}"
    )

if not evaluation_results:
    raise ValueError("No valid hierarchical clustering results were produced.")

evaluation_df = pd.DataFrame(evaluation_results)

evaluation_path = OUTPUT_DIR / "hierarchical_cluster_evaluation.csv"
evaluation_df.to_csv(evaluation_path, index=False)

# --------------------------------------------------
# 4. SELECT PROVISIONAL BEST K
# --------------------------------------------------

# Highest silhouette score is used as the primary
# selection criterion. Other metrics are also retained
# for interpretation.
best_row = evaluation_df.loc[
    evaluation_df["Silhouette_Score"].idxmax()
]

BEST_K = int(best_row["Requested_K"])
best_labels = cluster_assignments[BEST_K]

print("\n========== PROVISIONAL BEST RESULT ==========")
print("Selected K:", BEST_K)
print("Silhouette:", round(best_row["Silhouette_Score"], 4))
print("Davies-Bouldin:", round(best_row["Davies_Bouldin_Index"], 4))
print("Calinski-Harabasz:", round(best_row["Calinski_Harabasz_Score"], 2))

# --------------------------------------------------
# 5. SAVE BUYER CLUSTER ASSIGNMENTS
# --------------------------------------------------

assignments_df = pd.DataFrame({
    "client_id": buyer_ids,
    "hierarchical_cluster": best_labels
})

assignments_path = OUTPUT_DIR / "hierarchical_buyer_assignments.csv"
assignments_df.to_csv(assignments_path, index=False)

# Save cluster sizes.
cluster_sizes = (
    assignments_df["hierarchical_cluster"]
    .value_counts()
    .sort_index()
    .rename_axis("hierarchical_cluster")
    .reset_index(name="buyer_count")
)

cluster_sizes["buyer_percentage"] = (
    cluster_sizes["buyer_count"] / len(assignments_df) * 100
).round(2)

cluster_sizes.to_csv(
    OUTPUT_DIR / "hierarchical_cluster_sizes.csv",
    index=False
)

print("\n========== SELECTED CLUSTER SIZES ==========")
print(cluster_sizes.to_string(index=False))

# --------------------------------------------------
# 6. CREATE NUMERIC CLUSTER PROFILE
# --------------------------------------------------

if PROFILE_FILE.exists():

    profile_df = pd.read_csv(PROFILE_FILE)

    if "client_id" in profile_df.columns:
        profile_df = profile_df.drop_duplicates(subset=["client_id"])

        merged_df = profile_df.merge(
            assignments_df,
            on="client_id",
            how="inner",
            validate="one_to_one"
        )

        numeric_columns = merged_df.select_dtypes(
            include=np.number
        ).columns.tolist()

        # Do not calculate averages of IDs or cluster labels.
        excluded_columns = {
            "client_id",
            "hierarchical_cluster"
        }

        numeric_columns = [
            col for col in numeric_columns
            if col not in excluded_columns
        ]

        numeric_profile = (
            merged_df.groupby("hierarchical_cluster")[numeric_columns]
            .mean()
            .round(2)
            .reset_index()
        )

        numeric_profile.to_csv(
            OUTPUT_DIR / "hierarchical_numeric_cluster_profile.csv",
            index=False
        )

        print("\nNumeric cluster profile saved.")

    else:
        print("\nWarning: client_id missing from profile dataset.")
else:
    print("\nWarning: cluster_profile_data.csv not found. Profile skipped.")

# --------------------------------------------------
# 7. DENDROGRAM
# --------------------------------------------------

print("\n========== CREATING DENDROGRAM ==========")

plt.figure(figsize=(14, 7))

dendrogram(
    Z,
    truncate_mode="lastp",
    p=30,
    show_contracted=True,
    leaf_rotation=45,
    leaf_font_size=9
)

plt.title("Hierarchical Clustering Dendrogram (Truncated)")
plt.xlabel("Merged Buyer Groups")
plt.ylabel("Ward Linkage Distance")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "01_hierarchical_dendrogram.png",
    dpi=300
)
plt.close()

# --------------------------------------------------
# 8. EVALUATION GRAPHS
# --------------------------------------------------

plt.figure(figsize=(9, 5))
plt.plot(
    evaluation_df["Requested_K"],
    evaluation_df["Silhouette_Score"],
    marker="o"
)
plt.title("Hierarchical Clustering - Silhouette Scores")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.xticks(list(K_VALUES))
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "02_hierarchical_silhouette_scores.png",
    dpi=300
)
plt.close()

plt.figure(figsize=(9, 5))
plt.plot(
    evaluation_df["Requested_K"],
    evaluation_df["Davies_Bouldin_Index"],
    marker="o"
)
plt.title("Hierarchical Clustering - Davies-Bouldin Index")
plt.xlabel("Number of Clusters")
plt.ylabel("Davies-Bouldin Index")
plt.xticks(list(K_VALUES))
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "03_hierarchical_davies_bouldin_scores.png",
    dpi=300
)
plt.close()

plt.figure(figsize=(9, 5))
plt.plot(
    evaluation_df["Requested_K"],
    evaluation_df["Calinski_Harabasz_Score"],
    marker="o"
)
plt.title("Hierarchical Clustering - Calinski-Harabasz Scores")
plt.xlabel("Number of Clusters")
plt.ylabel("Calinski-Harabasz Score")
plt.xticks(list(K_VALUES))
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "04_hierarchical_calinski_harabasz_scores.png",
    dpi=300
)
plt.close()

# --------------------------------------------------
# 9. FINAL VALIDATION
# --------------------------------------------------

print("\n========== FINAL VALIDATION ==========")

print("Original buyers:", len(prepared_df))
print("Assigned buyers:", len(assignments_df))
print("Unique buyer IDs:", assignments_df["client_id"].nunique())
print("Unique hierarchical clusters:",
      assignments_df["hierarchical_cluster"].nunique())
print("Missing cluster assignments:",
      assignments_df["hierarchical_cluster"].isna().sum())
print("Duplicate buyer IDs:",
      assignments_df["client_id"].duplicated().sum())

assert len(assignments_df) == len(prepared_df)
assert assignments_df["client_id"].nunique() == len(prepared_df)
assert assignments_df["hierarchical_cluster"].notna().all()

print("\n========== STEP 5D COMPLETED ==========")
print("Output folder:", OUTPUT_DIR)
print("Evaluation file:", evaluation_path)
print("Assignment file:", assignments_path)
print("Selected cluster count:", BEST_K)
