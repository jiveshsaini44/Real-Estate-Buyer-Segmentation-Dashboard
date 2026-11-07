
from pathlib import Path
import pandas as pd

# ==========================================
# STEP 2: DATA INTEGRATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

CLIENTS_FILE = OUTPUT_DIR / "clients_cleaned.csv"
PROPERTIES_FILE = OUTPUT_DIR / "properties_cleaned.csv"

# ------------------------------------------
# 1. Load cleaned datasets
# ------------------------------------------

clients = pd.read_csv(CLIENTS_FILE)
properties = pd.read_csv(PROPERTIES_FILE)

print("\n========== DATASET INFORMATION ==========")

print("Clients:", clients.shape)
print("Properties:", properties.shape)

# ------------------------------------------
# 2. Standardize buyer reference columns
# ------------------------------------------

clients["client_id"] = (
    clients["client_id"].astype("string").str.strip()
)

properties["client_ref"] = (
    properties["client_ref"].astype("string").str.strip()
)

# Convert empty or textual missing values to NA
properties["client_ref"] = properties["client_ref"].replace(
    ["", "nan", "None", "NA", "<NA>"], pd.NA
)

# ------------------------------------------
# 3. Inspect listing status
# ------------------------------------------

print("\n========== LISTING STATUS ==========")

print(properties["listing_status"].value_counts(dropna=False))

# ------------------------------------------
# 4. Identify property records linked to clients
# ------------------------------------------

linked_properties = properties[
    properties["client_ref"].notna()
].copy()

unlinked_properties = properties[
    properties["client_ref"].isna()
].copy()

print("\nProperties linked to clients:", len(linked_properties))
print("Properties without client reference:", len(unlinked_properties))

# ------------------------------------------
# 5. Match property records to client IDs
# ------------------------------------------

matched_properties = linked_properties.merge(
    clients[["client_id"]],
    left_on="client_ref",
    right_on="client_id",
    how="left",
    indicator=True
)

matched_properties["client_match_status"] = (
    matched_properties["_merge"].map({
        "both": "Matched",
        "left_only": "Unmatched"
    })
)

print("\n========== CLIENT MATCHING ==========")

print(
    matched_properties["client_match_status"]
    .value_counts()
)

# Keep only property records belonging to known clients
matched_properties = matched_properties[
    matched_properties["_merge"] == "both"
].copy()

matched_properties.drop(
    columns=["_merge", "client_match_status"],
    inplace=True
)

# ------------------------------------------
# 6. Create property summary per buyer
# ------------------------------------------

# Only properties with a valid client match are summarized.
# Listing status is preserved for later analysis.

property_summary = (
    matched_properties
    .groupby("client_id")
    .agg(
        property_count=("listing_id", "nunique"),
        total_property_value=("sale_price", "sum"),
        average_property_price=("sale_price", "mean"),
        average_property_area=("floor_area_sqft", "mean"),
        property_categories=("unit_category", "nunique")
    )
    .reset_index()
)

# ------------------------------------------
# 7. Merge buyer information and property summary
# ------------------------------------------

buyer_integrated = clients.merge(
    property_summary,
    on="client_id",
    how="left"
)

# Buyers without matched property records
# should have zero property count.
buyer_integrated["property_count"] = (
    buyer_integrated["property_count"].fillna(0)
)

buyer_integrated["property_categories"] = (
    buyer_integrated["property_categories"].fillna(0)
)

# Do not replace missing property values with zero:
# missing means no matched property information.

# ------------------------------------------
# 8. Add buyer age
# ------------------------------------------

buyer_integrated["date_of_birth"] = pd.to_datetime(
    buyer_integrated["date_of_birth"],
    errors="coerce"
)

# Reference date: latest available transaction date.
# This avoids using today's age for historical transactions.

transaction_dates = pd.to_datetime(
    properties["transaction_date"],
    errors="coerce"
)

reference_date = transaction_dates.max()

buyer_integrated["age"] = (
    reference_date.year
    - buyer_integrated["date_of_birth"].dt.year
    - (
        (
            buyer_integrated["date_of_birth"].dt.month
            > reference_date.month
        )
        |
        (
            (
                buyer_integrated["date_of_birth"].dt.month
                == reference_date.month
            )
            &
            (
                buyer_integrated["date_of_birth"].dt.day
                > reference_date.day
            )
        )
    )
)

# ------------------------------------------
# 9. Save integrated datasets
# ------------------------------------------

buyer_integrated.to_csv(
    OUTPUT_DIR / "buyer_integrated.csv",
    index=False
)

matched_properties.to_csv(
    OUTPUT_DIR / "matched_properties.csv",
    index=False
)

unlinked_properties.to_csv(
    OUTPUT_DIR / "unlinked_properties.csv",
    index=False
)

# ------------------------------------------
# 10. Final report
# ------------------------------------------

print("\n========== INTEGRATION COMPLETED ==========")

print("Integrated buyer dataset:", buyer_integrated.shape)

print("\nBuyers with property records:")
print(
    (buyer_integrated["property_count"] > 0).sum()
)

print("\nBuyers without property records:")
print(
    (buyer_integrated["property_count"] == 0).sum()
)

print("\nIntegrated dataset missing values:")
print(buyer_integrated.isnull().sum())

print("\nSaved files:")
print("outputs/buyer_integrated.csv")
print("outputs/matched_properties.csv")
print("outputs/unlinked_properties.csv")
