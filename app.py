
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Real Estate Buyer Segmentation",
    page_icon="🏘️",
    layout="wide",
)

PROJECT_DIR = Path(__file__).resolve().parent

BUYER_FILE = PROJECT_DIR / "outputs" / "buyer_integrated.csv"

ASSIGNMENT_FILES = [
    PROJECT_DIR / "outputs" / "buyer_segmentation"
    / "kmeans_buyer_assignments.csv",

    PROJECT_DIR / "outputs" / "buyer_segmentation"
    / "kmeans_segmented_buyers.csv",
]


# ============================================================
# CLUSTER INTERPRETATION
# ============================================================

CLUSTER_NAMES = {
    "0": "Cluster 0 — Lower-Average-Price, Smaller-Area Buyers",
    "1": "Cluster 1 — Higher-Average-Price, Larger-Area Buyers",
}


def cluster_name(cluster):
    cluster = str(cluster)
    return CLUSTER_NAMES.get(cluster, f"Cluster {cluster}")


def cluster_color(cluster):
    cluster = str(cluster)

    if cluster == "0":
        return [52, 152, 219, 190]

    if cluster == "1":
        return [231, 76, 60, 190]

    return [120, 120, 120, 190]


# ============================================================
# COLUMN NORMALIZATION
# ============================================================

def normalize_column_name(name):
    """Convert column names into a consistent format."""
    name = str(name).strip().lower()
    name = name.replace(" ", "_")
    name = name.replace("-", "_")
    name = name.replace("/", "_")
    name = name.replace("(", "")
    name = name.replace(")", "")
    name = name.replace("%", "percent")
    return name


def normalize_columns(df):
    df = df.copy()
    df.columns = [normalize_column_name(c) for c in df.columns]

    # Common alternate column names
    aliases = {
        "client_id": [
            "client_id",
            "buyer_id",
            "customer_id",
            "id",
        ],
        "cluster": [
            "cluster",
            "cluster_label",
            "kmeans_cluster",
            "kmeans_label",
            "segment",
            "cluster_number",
        ],
        "country": ["country", "client_country"],
        "region": ["region", "state", "province", "state_province"],
        "client_type": ["client_type", "buyer_type"],
        "acquisition_purpose": [
            "acquisition_purpose",
            "purpose",
            "purchase_purpose",
        ],
        "age": ["age", "buyer_age"],
        "satisfaction_score": [
            "satisfaction_score",
            "customer_satisfaction",
            "satisfaction",
        ],
        "property_count": [
            "property_count",
            "number_of_properties",
            "properties_owned",
        ],
        "total_property_value": [
            "total_property_value",
            "total_value",
        ],
        "average_property_price": [
            "average_property_price",
            "avg_property_price",
            "average_price",
        ],
        "average_property_area": [
            "average_property_area",
            "avg_property_area",
            "average_area",
        ],
        "property_categories": [
            "property_categories",
            "number_of_property_categories",
        ],
        "loan_applied": ["loan_applied", "loan_status"],
        "referral_channel": ["referral_channel", "referral"],
        "gender": ["gender", "sex"],
    }

    for canonical, possible_names in aliases.items():
        if canonical in df.columns:
            continue

        for possible in possible_names:
            possible = normalize_column_name(possible)

            if possible in df.columns:
                df = df.rename(columns={possible: canonical})
                break

    return df


# ============================================================
# LOAD BUYER DATA AND CLUSTER ASSIGNMENTS
# ============================================================

@st.cache_data
def load_data():
    if not BUYER_FILE.exists():
        raise FileNotFoundError(
            f"Buyer dataset not found:\n{BUYER_FILE}"
        )

    buyers = pd.read_csv(BUYER_FILE)
    buyers = normalize_columns(buyers)

    if "client_id" not in buyers.columns:
        raise ValueError(
            "buyer_integrated.csv must contain client_id."
        )

    buyers["client_id"] = buyers["client_id"].astype(str).str.strip()

    assignment_df = None
    assignment_path = None

    for file_path in ASSIGNMENT_FILES:
        if not file_path.exists():
            continue

        candidate = pd.read_csv(file_path)
        candidate = normalize_columns(candidate)

        if "client_id" in candidate.columns and "cluster" in candidate.columns:
            assignment_df = candidate
            assignment_path = file_path
            break

    if assignment_df is None:
        raise FileNotFoundError(
            "Could not find a valid K-Means assignment file.\n\n"
            "Expected one of:\n"
            + "\n".join(str(p) for p in ASSIGNMENT_FILES)
            + "\n\nEach file must contain client_id and cluster."
        )

    assignment_df["client_id"] = (
        assignment_df["client_id"].astype(str).str.strip()
    )

    assignment_df["cluster"] = (
        assignment_df["cluster"].astype(str).str.strip()
    )

    # Remove missing cluster labels and duplicate IDs.
    assignment_df = assignment_df.dropna(subset=["cluster"])
    assignment_df = assignment_df[
        ["client_id", "cluster"]
    ].drop_duplicates(subset=["client_id"])

    # The integrated dataset is the source for buyer characteristics.
    # Cluster labels come from the saved K-Means assignment output.
    data = buyers.merge(
        assignment_df,
        on="client_id",
        how="inner",
        validate="one_to_one",
    )

    if data.empty:
        raise ValueError(
            "No buyer IDs matched between buyer_integrated.csv "
            "and the saved cluster assignments."
        )

    # Convert expected numeric columns safely.
    numeric_columns = [
        "age",
        "satisfaction_score",
        "property_count",
        "total_property_value",
        "average_property_price",
        "average_property_area",
        "property_categories",
    ]

    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(
                data[column], errors="coerce"
            )

    return data, assignment_path


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def available_values(df, column):
    if column not in df.columns:
        return []

    values = (
        df[column]
        .dropna()
        .astype(str)
        .str.strip()
    )

    values = values[values != ""]
    return sorted(values.unique().tolist())


def apply_filter(df, column, selected_values):
    if column not in df.columns:
        return df

    if not selected_values:
        return df

    return df[df[column].astype(str).isin(selected_values)]


def format_number(value, decimals=2):
    if pd.isna(value):
        return "N/A"

    return f"{value:,.{decimals}f}"


def display_cluster_label(cluster):
    return cluster_name(cluster)


def get_numeric_columns(df):
    preferred = [
        "age",
        "satisfaction_score",
        "property_count",
        "total_property_value",
        "average_property_price",
        "average_property_area",
        "property_categories",
    ]

    return [
        col for col in preferred
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
    ]


def show_missing_columns(df, columns):
    missing = [column for column in columns if column not in df.columns]

    if missing:
        st.warning(
            "Some optional fields are unavailable in the dataset: "
            + ", ".join(missing)
        )


# ============================================================
# GEOGRAPHIC COORDINATES
# ============================================================

# Coordinates are approximate geographic centroids.
# Region coordinates are used for US states when available.
US_STATE_COORDS = {
    "Alabama": (32.806671, -86.791130),
    "Alaska": (61.370716, -152.404419),
    "Arizona": (33.729759, -111.431221),
    "Arkansas": (34.969704, -92.373123),
    "California": (36.116203, -119.681564),
    "Colorado": (39.059811, -105.311104),
    "Connecticut": (41.597782, -72.755371),
    "Delaware": (39.318523, -75.507141),
    "Florida": (27.766279, -81.686783),
    "Georgia": (33.040619, -83.643074),
    "Hawaii": (21.094318, -157.498337),
    "Idaho": (44.240459, -114.478828),
    "Illinois": (40.349457, -88.986137),
    "Indiana": (39.849426, -86.258278),
    "Iowa": (42.011539, -93.210526),
    "Kansas": (38.526600, -96.726486),
    "Kentucky": (37.668140, -84.670067),
    "Louisiana": (31.169546, -91.867805),
    "Maine": (44.693947, -69.381927),
    "Maryland": (39.063946, -76.802101),
    "Massachusetts": (42.230171, -71.530106),
    "Michigan": (43.326618, -84.536095),
    "Minnesota": (45.694454, -93.900192),
    "Mississippi": (32.741646, -89.678696),
    "Missouri": (38.456085, -92.288368),
    "Montana": (46.921925, -110.454353),
    "Nebraska": (41.125370, -98.268082),
    "Nevada": (38.313515, -117.055374),
    "New Hampshire": (43.452492, -71.563896),
    "New Jersey": (40.298904, -74.521011),
    "New Mexico": (34.840515, -106.248482),
    "New York": (42.165726, -74.948051),
    "North Carolina": (35.630066, -79.806419),
    "North Dakota": (47.528912, -99.784012),
    "Ohio": (40.388783, -82.764915),
    "Oklahoma": (35.565342, -96.928917),
    "Oregon": (44.572021, -122.070938),
    "Pennsylvania": (40.590752, -77.209755),
    "Rhode Island": (41.680893, -71.511780),
    "South Carolina": (33.856892, -80.945007),
    "South Dakota": (44.299782, -99.438828),
    "Tennessee": (35.747845, -86.692345),
    "Texas": (31.054487, -97.563461),
    "Utah": (40.150032, -111.862434),
    "Vermont": (44.045876, -72.710686),
    "Virginia": (37.769337, -78.169968),
    "Washington": (47.400902, -121.490494),
    "West Virginia": (38.491226, -80.954453),
    "Wisconsin": (44.268543, -89.616508),
    "Wyoming": (42.755966, -107.302490),
}

COUNTRY_COORDS = {
    "United States": (39.5, -98.35),
    "USA": (39.5, -98.35),
    "US": (39.5, -98.35),
    "United Kingdom": (55.3781, -3.4360),
    "UK": (55.3781, -3.4360),
    "Canada": (56.1304, -106.3468),
    "Germany": (51.1657, 10.4515),
    "France": (46.2276, 2.2137),
    "Belgium": (50.5039, 4.4699),
    "Mexico": (23.6345, -102.5528),
    "Australia": (-25.2744, 133.7751),
    "Russia": (61.5240, 105.3188),
    "Denmark": (56.2639, 9.5018),
}


def find_region_coordinates(region, country):
    region = str(region).strip()
    country = str(country).strip()

    # Match US state names case-insensitively.
    for state, coords in US_STATE_COORDS.items():
        if state.lower() == region.lower():
            return coords

    # Fall back to country centroid for non-US regions.
    for name, coords in COUNTRY_COORDS.items():
        if name.lower() == country.lower():
            return coords

    return None


# ============================================================
# LOAD APPLICATION DATA
# ============================================================

try:
    data, assignment_path = load_data()

except Exception as error:
    st.error(f"Unable to load the project data: {error}")

    st.info(
        "Check that buyer_integrated.csv and the saved K-Means "
        "assignment CSV are present in the outputs folder."
    )

    st.stop()


# ============================================================
# APP HEADER
# ============================================================

st.title("🏘️ Real Estate Buyer Segmentation Dashboard")

st.markdown(
    """
    Explore buyer clusters discovered by the saved **K-Means model**.
    The dashboard uses the integrated buyer dataset and saved cluster
    assignments from the project.
    """
)

st.caption(
    f"Buyer data: {BUYER_FILE.name} | "
    f"Cluster assignments: {assignment_path.name}"
)

st.divider()


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.title("Dashboard Filters")

st.sidebar.caption(
    "Filters update the overview, investor behavior, geography, "
    "and cluster insights."
)

filter_columns = [
    ("country", "Country"),
    ("region", "Region"),
    ("acquisition_purpose", "Acquisition Purpose"),
    ("client_type", "Client Type"),
]

selected_filters = {}

for column, label in filter_columns:
    values = available_values(data, column)

    if values:
        selected_filters[column] = st.sidebar.multiselect(
            label,
            options=values,
            default=values,
            key=f"filter_{column}",
        )
    else:
        selected_filters[column] = []
        st.sidebar.warning(f"{label} column not found.")

filtered_data = data.copy()

for column, selected_values in selected_filters.items():
    filtered_data = apply_filter(
        filtered_data,
        column,
        selected_values,
    )


if filtered_data.empty:
    st.warning(
        "No buyers match the selected filters. "
        "Please broaden your filter selection."
    )
    st.stop()


# ============================================================
# GLOBAL KPI SUMMARY
# ============================================================

st.subheader("Filtered Buyer Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Buyers",
    f"{filtered_data['client_id'].nunique():,}",
)

col2.metric(
    "Clusters",
    f"{filtered_data['cluster'].nunique():,}",
)

if "average_property_price" in filtered_data.columns:
    col3.metric(
        "Average Property Price",
        format_number(filtered_data["average_property_price"].mean()),
    )
else:
    col3.metric("Average Property Price", "N/A")

if "property_count" in filtered_data.columns:
    col4.metric(
        "Average Properties per Buyer",
        format_number(filtered_data["property_count"].mean()),
    )
else:
    col4.metric("Average Properties per Buyer", "N/A")

st.divider()


# ============================================================
# SECTION 1: BUYER SEGMENTATION OVERVIEW
# ============================================================

st.header("1. Buyer Segmentation Overview")

cluster_counts = (
    filtered_data.groupby("cluster")["client_id"]
    .nunique()
    .sort_index()
)

cluster_distribution = cluster_counts.rename("Buyer Count").to_frame()
cluster_distribution["Percentage"] = (
    cluster_distribution["Buyer Count"]
    / cluster_distribution["Buyer Count"].sum()
    * 100
)

cluster_distribution["Buyer Segment"] = [
    display_cluster_label(cluster)
    for cluster in cluster_distribution.index
]

left, right = st.columns(2)

with left:
    st.markdown("#### Cluster Distribution")

    chart_data = cluster_distribution[
        ["Buyer Count"]
    ].copy()

    chart_data.index = cluster_distribution["Buyer Segment"]

    st.bar_chart(chart_data)

with right:
    st.markdown("#### Cluster Distribution Details")

    display_distribution = cluster_distribution[
        ["Buyer Segment", "Buyer Count", "Percentage"]
    ].copy()

    display_distribution["Percentage"] = (
        display_distribution["Percentage"].round(2)
    )

    st.dataframe(
        display_distribution.reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )

st.caption(
    "Cluster percentages are calculated from the currently filtered buyers."
)


# ============================================================
# SECTION 2: INVESTOR BEHAVIOR DASHBOARD
# ============================================================

st.header("2. Investor Behavior Dashboard")

purpose_col, loan_col = st.columns(2)

with purpose_col:
    st.markdown("#### Acquisition Purpose by Cluster")

    if "acquisition_purpose" in filtered_data.columns:
        purpose_table = pd.crosstab(
            filtered_data["cluster"],
            filtered_data["acquisition_purpose"],
        )

        purpose_table.index = [
            display_cluster_label(cluster)
            for cluster in purpose_table.index
        ]

        st.bar_chart(purpose_table)

        st.dataframe(
            purpose_table,
            use_container_width=True,
        )

    else:
        st.warning("Acquisition purpose data is unavailable.")

with loan_col:
    st.markdown("#### Loan Application by Cluster")

    if "loan_applied" in filtered_data.columns:
        loan_table = pd.crosstab(
            filtered_data["cluster"],
            filtered_data["loan_applied"],
        )

        loan_table.index = [
            display_cluster_label(cluster)
            for cluster in loan_table.index
        ]

        st.bar_chart(loan_table)

        st.dataframe(
            loan_table,
            use_container_width=True,
        )

    else:
        st.warning("Loan application data is unavailable.")


st.markdown("#### Investment Purpose Buyer Share")

if "acquisition_purpose" in filtered_data.columns:
    investment_data = filtered_data[
        filtered_data["acquisition_purpose"]
        .astype(str)
        .str.lower()
        .str.contains("investment", na=False)
    ]

    investment_share = (
        investment_data.groupby("cluster")["client_id"]
        .nunique()
        .div(
            filtered_data.groupby("cluster")["client_id"].nunique()
        )
        .mul(100)
        .fillna(0)
    )

    investment_share_df = investment_share.rename(
        "Investment Purpose Buyers (%)"
    ).to_frame()

    investment_share_df.index = [
        display_cluster_label(cluster)
        for cluster in investment_share_df.index
    ]

    st.bar_chart(investment_share_df)

    st.caption(
        "Investment-purpose share = buyers whose acquisition purpose "
        "contains 'Investment' divided by all buyers in that cluster."
    )


# ============================================================
# SECTION 3: GEOGRAPHIC BUYER ANALYSIS
# ============================================================

st.header("3. Geographic Buyer Analysis")

geo_col1, geo_col2 = st.columns(2)

with geo_col1:
    st.markdown("#### Buyers by Region")

    if "region" in filtered_data.columns:
        region_counts = (
            filtered_data.groupby("region")["client_id"]
            .nunique()
            .sort_values(ascending=False)
            .head(15)
        )

        st.bar_chart(region_counts)

    else:
        st.warning("Region data is unavailable.")

with geo_col2:
    st.markdown("#### Buyers by Country")

    if "country" in filtered_data.columns:
        country_counts = (
            filtered_data.groupby("country")["client_id"]
            .nunique()
            .sort_values(ascending=False)
            .head(15)
        )

        st.bar_chart(country_counts)

    else:
        st.warning("Country data is unavailable.")


st.markdown("#### Buyer Segment Map")

if "region" in filtered_data.columns and "country" in filtered_data.columns:

    geographic_rows = []

    geo_grouped = (
        filtered_data.groupby(
            ["cluster", "country", "region"],
            dropna=False,
        )["client_id"]
        .nunique()
        .reset_index(name="buyer_count")
    )

    for _, row in geo_grouped.iterrows():
        if pd.isna(row["region"]) or pd.isna(row["country"]):
            continue

        coordinates = find_region_coordinates(
            row["region"],
            row["country"],
        )

        if coordinates is None:
            continue

        latitude, longitude = coordinates

        # Slight offset helps visually distinguish clusters
        # that share the same regional centroid.
        cluster_value = str(row["cluster"])

        if cluster_value == "0":
            latitude -= 0.15
            longitude -= 0.15
        elif cluster_value == "1":
            latitude += 0.15
            longitude += 0.15

        geographic_rows.append(
            {
                "cluster": cluster_value,
                "country": row["country"],
                "region": row["region"],
                "buyer_count": int(row["buyer_count"]),
                "latitude": latitude,
                "longitude": longitude,
                "color": cluster_color(cluster_value),
                "radius": max(
                    30000,
                    int(row["buyer_count"]) * 500,
                ),
            }
        )

    map_data = pd.DataFrame(geographic_rows)

    if not map_data.empty:

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_data,
            get_position="[longitude, latitude]",
            get_fill_color="color",
            get_radius="radius",
            radius_min_pixels=5,
            radius_max_pixels=30,
            pickable=True,
            auto_highlight=True,
        )

        view_state = pdk.ViewState(
            latitude=39.5,
            longitude=-98.35,
            zoom=1.5,
            pitch=0,
        )

        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip={
                "html": (
                    "<b>Cluster:</b> {cluster}<br/>"
                    "<b>Country:</b> {country}<br/>"
                    "<b>Region:</b> {region}<br/>"
                    "<b>Buyer Count:</b> {buyer_count}"
                ),
                "style": {
                    "backgroundColor": "white",
                    "color": "black",
                },
            },
        )

        st.pydeck_chart(
            deck,
            use_container_width=True,
        )

        st.caption(
            "Map points use approximate regional or country centroids. "
            "They represent buyer counts, not individual buyer locations."
        )

        st.markdown(
            """
            **Cluster colors**
            - 🔵 Cluster 0
            - 🔴 Cluster 1
            """
        )

        mapped_regions = set(map_data["region"].astype(str))
        all_regions = set(
            filtered_data["region"].dropna().astype(str)
        )

        unmapped_regions = sorted(all_regions - mapped_regions)

        if unmapped_regions:
            st.info(
                "Some regions were not mapped because coordinates "
                "were unavailable: "
                + ", ".join(unmapped_regions[:20])
            )

    else:
        st.warning(
            "No recognized geographic coordinates were found for "
            "the selected regions."
        )

else:
    st.warning(
        "Country and region columns are required for geographic analysis."
    )


# ============================================================
# SECTION 4: SEGMENT INSIGHTS PANEL
# ============================================================

st.header("4. Segment Insights Panel")

available_clusters = sorted(
    filtered_data["cluster"].astype(str).unique().tolist()
)

selected_cluster = st.selectbox(
    "Select a cluster to inspect",
    options=available_clusters,
    format_func=display_cluster_label,
)

segment_data = filtered_data[
    filtered_data["cluster"].astype(str) == selected_cluster
].copy()

st.subheader(display_cluster_label(selected_cluster))

segment_col1, segment_col2, segment_col3, segment_col4 = st.columns(4)

segment_col1.metric(
    "Buyers in Cluster",
    f"{segment_data['client_id'].nunique():,}",
)

if "average_property_price" in segment_data.columns:
    segment_col2.metric(
        "Average Property Price",
        format_number(segment_data["average_property_price"].mean()),
    )
else:
    segment_col2.metric("Average Property Price", "N/A")

if "average_property_area" in segment_data.columns:
    segment_col3.metric(
        "Average Property Area",
        format_number(segment_data["average_property_area"].mean()),
    )
else:
    segment_col3.metric("Average Property Area", "N/A")

if "property_count" in segment_data.columns:
    segment_col4.metric(
        "Average Properties per Buyer",
        format_number(segment_data["property_count"].mean()),
    )
else:
    segment_col4.metric("Average Properties per Buyer", "N/A")


st.markdown("#### Numeric Cluster Profile")

numeric_columns = get_numeric_columns(segment_data)

if numeric_columns:
    numeric_profile = (
        segment_data[numeric_columns]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .T
    )

    numeric_profile.columns = [
        "Count",
        "Mean",
        "Median",
        "Standard Deviation",
        "Minimum",
        "Maximum",
    ]

    st.dataframe(
        numeric_profile.round(2),
        use_container_width=True,
    )
else:
    st.warning("No numeric profiling columns were found.")


st.markdown("#### Categorical Cluster Profile")

categorical_columns = [
    "country",
    "region",
    "client_type",
    "acquisition_purpose",
    "loan_applied",
    "referral_channel",
]

for column in categorical_columns:
    if column not in segment_data.columns:
        continue

    with st.expander(f"{column.replace('_', ' ').title()} distribution"):
        category_counts = (
            segment_data[column]
            .fillna("Unknown")
            .astype(str)
            .value_counts()
        )

        category_percentages = (
            category_counts / category_counts.sum() * 100
        )

        category_table = pd.DataFrame(
            {
                "Buyer Count": category_counts,
                "Percentage": category_percentages.round(2),
            }
        )

        st.dataframe(
            category_table,
            use_container_width=True,
        )


# ============================================================
# FOOTER / DATA VALIDATION
# ============================================================

st.divider()

with st.expander("Data and model information"):
    st.write("**Buyer source:**", str(BUYER_FILE))
    st.write("**Cluster assignment source:**", str(assignment_path))
    st.write("**Buyers in source dataset:**", f"{len(data):,}")
    st.write(
        "**Unique buyer IDs:**",
        f"{data['client_id'].nunique():,}",
    )
    st.write(
        "**Cluster labels found:**",
        sorted(data["cluster"].astype(str).unique().tolist()),
    )

    if data["client_id"].duplicated().any():
        st.warning(
            "Duplicate client IDs were detected after loading. "
            "Check the data integration and assignment files."
        )
    else:
        st.success("No duplicate buyer IDs detected.")

    if data["cluster"].isna().any():
        st.warning("Some buyers have missing cluster labels.")
    else:
        st.success("All loaded buyers have cluster assignments.")

st.caption(
    "Buyer segmentation is based on the saved K-Means K=2 results. "
    "Cluster descriptions are descriptive, not income or wealth classifications."
)
