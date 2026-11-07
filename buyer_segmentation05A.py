
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)


# ==========================================================
# STEP 5A: BUYER SEGMENTATION USING K-MEANS
# ==========================================================

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "outputs" / "clustering_preparation"
OUTPUT_DIR = BASE_DIR / "outputs" / "buyer_segmentation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREPARED_FILE = INPUT_DIR / "clustering_input_prepared.csv"
ORIGINAL_FILE = BASE_DIR / "outputs" / "buyer_integrated.csv"


print("\n========== STEP 5A: K-MEANS BUYER SEGMENTATION ==========")


# ----------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------

prepared_df = pd.read_csv(PREPARED_FILE)
original_df = pd.read_csv(ORIGINAL_FILE)

if "client_id" not in prepared_df.columns:
    raise ValueError("client_id is missing from prepared dataset.")

if prepared_df["client_id"].duplicated().any():
    raise ValueError("Duplicate buyer IDs found.")

if prepared_df["client_id"].nunique() != len(original_df):
    raise ValueError("Buyer count does not match original dataset.")

# Keep buyer IDs separate from clustering features.
X = prepared_df.drop(columns=["client_id"])

# Ensure all clustering features are numeric.
X = X.apply(pd.to_numeric, errors="coerce")

if X.isna().any().any():
    raise ValueError("Missing or non-numeric values found in clustering data.")

if not np.isfinite(X.to_numpy()).all():
    raise ValueError("Infinite values found in clustering data.")

print("\nDataset loaded successfully.")
print("Total buyers:", len(X))
print("Clustering features:", X.shape[1])


# ----------------------------------------------------------
# 2. TEST DIFFERENT NUMBERS OF CLUSTERS
# ----------------------------------------------------------

MIN_CLUSTERS = 2
MAX_CLUSTERS = 10

results = []

print("\nEvaluating different cluster counts...")

for k in range(MIN_CLUSTERS, MAX_CLUSTERS + 1):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    labels = model.fit_predict(X)

    silhouette = silhouette_score(X, labels)
    davies_bouldin = davies_bouldin_score(X, labels)
    calinski_harabasz = calinski_harabasz_score(X, labels)

    results.append({
        "Number_of_Clusters": k,
        "Inertia": model.inertia_,
        "Silhouette_Score": silhouette,
        "Davies_Bouldin_Index": davies_bouldin,
        "Calinski_Harabasz_Score": calinski_harabasz
    })

    print(
        f"K={k} | "
        f"Silhouette={silhouette:.4f} | "
        f"DBI={davies_bouldin:.4f} | "
        f"CH={calinski_harabasz:.2f}"
    )


# ----------------------------------------------------------
# 3. SAVE CLUSTER EVALUATION RESULTS
# ----------------------------------------------------------

evaluation_df = pd.DataFrame(results)

evaluation_df.to_csv(
    OUTPUT_DIR / "kmeans_cluster_evaluation.csv",
    index=False
)

# Highest silhouette score gives the provisional K.
best_row = evaluation_df.loc[
    evaluation_df["Silhouette_Score"].idxmax()
]

best_k = int(best_row["Number_of_Clusters"])

print("\n========== PROVISIONAL CLUSTER SELECTION ==========")
print("Selected number of clusters:", best_k)
print(
    "Best Silhouette Score:",
    round(best_row["Silhouette_Score"], 4)
)
print(
    "Davies-Bouldin Index:",
    round(best_row["Davies_Bouldin_Index"], 4)
)
print(
    "Calinski-Harabasz Score:",
    round(best_row["Calinski_Harabasz_Score"], 2)
)


# ----------------------------------------------------------
# 4. ELBOW METHOD GRAPH
# ----------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    evaluation_df["Number_of_Clusters"],
    evaluation_df["Inertia"],
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for K-Means")
plt.xticks(range(MIN_CLUSTERS, MAX_CLUSTERS + 1))
plt.grid(True)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "01_elbow_method.png",
    dpi=300
)

plt.close()


# ----------------------------------------------------------
# 5. SILHOUETTE SCORE GRAPH
# ----------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    evaluation_df["Number_of_Clusters"],
    evaluation_df["Silhouette_Score"],
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score for Different Cluster Counts")
plt.xticks(range(MIN_CLUSTERS, MAX_CLUSTERS + 1))
plt.grid(True)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "02_silhouette_scores.png",
    dpi=300
)

plt.close()


# ----------------------------------------------------------
# 6. FIT FINAL PROVISIONAL K-MEANS MODEL
# ----------------------------------------------------------

final_model = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=20
)

cluster_labels = final_model.fit_predict(X)

# Attach cluster assignments to buyer IDs.
assignments = pd.DataFrame({
    "client_id": prepared_df["client_id"],
    "Cluster": cluster_labels
})

assignments["Cluster"] = (
    assignments["Cluster"].astype(int)
)

assignments.to_csv(
    OUTPUT_DIR / "kmeans_buyer_assignments.csv",
    index=False
)


# ----------------------------------------------------------
# 7. CLUSTER SIZE ANALYSIS
# ----------------------------------------------------------

cluster_sizes = (
    assignments["Cluster"]
    .value_counts()
    .sort_index()
    .rename_axis("Cluster")
    .reset_index(name="Buyer_Count")
)

cluster_sizes["Percentage"] = (
    cluster_sizes["Buyer_Count"] / len(assignments) * 100
)

cluster_sizes.to_csv(
    OUTPUT_DIR / "kmeans_cluster_sizes.csv",
    index=False
)

print("\n========== CLUSTER SIZE DISTRIBUTION ==========")
print(cluster_sizes.to_string(index=False))


# ----------------------------------------------------------
# 8. COMBINE CLUSTERS WITH ORIGINAL BUYER DATA
# ----------------------------------------------------------

segmented_buyers = original_df.merge(
    assignments,
    on="client_id",
    how="inner",
    validate="one_to_one"
)

if len(segmented_buyers) != len(original_df):
    raise ValueError("Some buyers were lost during cluster merging.")

segmented_buyers.to_csv(
    OUTPUT_DIR / "kmeans_segmented_buyers.csv",
    index=False
)


# ----------------------------------------------------------
# 9. NUMERICAL CLUSTER PROFILING
# ----------------------------------------------------------

numeric_profile_columns = [
    "age",
    "satisfaction_score",
    "property_count",
    "total_property_value",
    "average_property_price",
    "average_property_area",
    "property_categories"
]

available_numeric = [
    col for col in numeric_profile_columns
    if col in segmented_buyers.columns
]

numeric_profile = (
    segmented_buyers
    .groupby("Cluster")[available_numeric]
    .mean()
    .round(2)
)

numeric_profile.to_csv(
    OUTPUT_DIR / "kmeans_numeric_cluster_profile.csv"
)


# ----------------------------------------------------------
# 10. CATEGORICAL CLUSTER PROFILING
# ----------------------------------------------------------

categorical_profile_columns = [
    "client_type",
    "acquisition_purpose",
    "loan_applied",
    "referral_channel",
    "country",
    "region"
]

for col in categorical_profile_columns:

    if col not in segmented_buyers.columns:
        continue

    profile = pd.crosstab(
        segmented_buyers["Cluster"],
        segmented_buyers[col],
        normalize="index"
    ) * 100

    profile = profile.round(2)

    profile.to_csv(
        OUTPUT_DIR / f"kmeans_{col}_profile.csv"
    )


# ----------------------------------------------------------
# 11. SAVE THE K-MEANS MODEL
# ----------------------------------------------------------

import joblib

joblib.dump(
    final_model,
    OUTPUT_DIR / "kmeans_buyer_model.joblib"
)


# ----------------------------------------------------------
# 12. FINAL VALIDATION
# ----------------------------------------------------------

print("\n========== FINAL VALIDATION ==========")

print("Original buyers:", len(original_df))
print("Segmented buyers:", len(segmented_buyers))
print("Unique assigned buyers:", assignments["client_id"].nunique())
print("Selected provisional K:", best_k)

print(
    "Missing cluster assignments:",
    assignments["Cluster"].isna().sum()
)

print(
    "Duplicate buyer assignments:",
    assignments["client_id"].duplicated().sum()
)

print("\nGenerated output files:")

for file in sorted(OUTPUT_DIR.iterdir()):
    print("-", file.name)

print("\nSTEP 5A K-MEANS SEGMENTATION COMPLETED SUCCESSFULLY!")
