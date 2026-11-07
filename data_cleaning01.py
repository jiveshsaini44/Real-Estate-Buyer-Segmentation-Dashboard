
from pathlib import Path
import pandas as pd
import numpy as np

# ==========================================
# STEP 1: DATA CLEANING
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

CLIENTS_FILE = DATA_DIR / "clients.csv"
PROPERTIES_FILE = DATA_DIR / "properties.csv"

# ------------------------------------------
# 1. Load datasets
# ------------------------------------------

clients = pd.read_csv(CLIENTS_FILE)
properties = pd.read_csv(PROPERTIES_FILE)

print("\n========== ORIGINAL DATA ==========")
print("Clients shape:", clients.shape)
print("Properties shape:", properties.shape)

# ------------------------------------------
# 2. Standardize column names
# ------------------------------------------

clients.columns = clients.columns.str.strip().str.lower()
properties.columns = properties.columns.str.strip().str.lower()

# ------------------------------------------
# 3. Clean text columns
# ------------------------------------------

def clean_text_columns(df):
    for column in df.select_dtypes(include="object").columns:
        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
            .replace("", pd.NA)
        )
    return df

clients = clean_text_columns(clients)
properties = clean_text_columns(properties)

# ------------------------------------------
# 4. Remove duplicate records
# ------------------------------------------

client_duplicates = clients.duplicated(subset="client_id").sum()
property_duplicates = properties.duplicated(subset="listing_id").sum()

print("\nDuplicate client IDs:", client_duplicates)
print("Duplicate listing IDs:", property_duplicates)

clients = clients.drop_duplicates(
    subset="client_id",
    keep="first"
)

properties = properties.drop_duplicates(
    subset="listing_id",
    keep="first"
)

# ------------------------------------------
# 5. Standardize categorical values
# ------------------------------------------

client_category_columns = [
    "client_type",
    "gender",
    "country",
    "region",
    "acquisition_purpose",
    "loan_applied",
    "referral_channel"
]

for column in client_category_columns:
    clients[column] = (
        clients[column]
        .astype("string")
        .str.strip()
        .str.title()
    )

properties["unit_category"] = (
    properties["unit_category"]
    .astype("string")
    .str.strip()
    .str.title()
)

properties["listing_status"] = (
    properties["listing_status"]
    .astype("string")
    .str.strip()
    .str.title()
)

# ------------------------------------------
# 6. Convert dates
# ------------------------------------------

clients["date_of_birth"] = pd.to_datetime(
    clients["date_of_birth"],
    format="mixed",
    dayfirst=True,
    errors="coerce"
)

properties["transaction_date"] = pd.to_datetime(
    properties["transaction_date"],
    format="mixed",
    dayfirst=True,
    errors="coerce"
)

# ------------------------------------------
# 7. Convert numeric columns
# ------------------------------------------

clients["satisfaction_score"] = pd.to_numeric(
    clients["satisfaction_score"],
    errors="coerce"
)

properties["floor_area_sqft"] = pd.to_numeric(
    properties["floor_area_sqft"],
    errors="coerce"
)

properties["tower_number"] = pd.to_numeric(
    properties["tower_number"],
    errors="coerce"
)

properties["unit_number"] = pd.to_numeric(
    properties["unit_number"],
    errors="coerce"
)

# Remove currency symbols and commas
properties["sale_price"] = (
    properties["sale_price"]
    .astype("string")
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
    .str.strip()
)

properties["sale_price"] = pd.to_numeric(
    properties["sale_price"],
    errors="coerce"
)

# ------------------------------------------
# 8. Validate values
# ------------------------------------------

# Invalid satisfaction scores
clients.loc[
    ~clients["satisfaction_score"].between(1, 5),
    "satisfaction_score"
] = np.nan

# Invalid property measurements
properties.loc[
    properties["floor_area_sqft"] <= 0,
    "floor_area_sqft"
] = np.nan

properties.loc[
    properties["sale_price"] <= 0,
    "sale_price"
] = np.nan

# ------------------------------------------
# 9. Check missing values
# ------------------------------------------

print("\n========== MISSING VALUES ==========")

print("\nClients:")
print(clients.isnull().sum())

print("\nProperties:")
print(properties.isnull().sum())

# ------------------------------------------
# 10. Save cleaned datasets
# ------------------------------------------

clients.to_csv(
    OUTPUT_DIR / "clients_cleaned.csv",
    index=False
)

properties.to_csv(
    OUTPUT_DIR / "properties_cleaned.csv",
    index=False
)

# ------------------------------------------
# 11. Final summary
# ------------------------------------------

print("\n========== CLEANING COMPLETED ==========")

print("Cleaned clients:", clients.shape)
print("Cleaned properties:", properties.shape)

print("\nFiles saved:")
print("outputs/clients_cleaned.csv")
print("outputs/properties_cleaned.csv")
