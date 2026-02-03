# app.py
import pandas as pd
import streamlit as st
import plotly.express as px

# ---- CONFIG: set these to match your VIW_FID_EPI.csv columns ----
DATE_COL = "MMWR_WEEKSTARTDATE"        # e.g. "date", "WEEK_START_DATE"
COUNTRY_COL = "COUNTRY_AREA_TERRITORY"  # e.g. "country", "COUNTRY"
CASES_COL = "ILI_CASE"      # e.g. "cases", "TOTAL_POSITIVE"
# ----------------------------------------------------------------

st.set_page_config(
    page_title="Influenza Case Explorer",
    layout="wide",
)

st.title("Influenza Case Explorer (VIW_FID_EPI.csv)")

st.markdown(
    "Upload or point to your **VIW_FID_EPI.csv** file, then examine influenza cases "
    "by country within any **3‑year window**."
)

# --- Load data ---
st.sidebar.header("Data input")

uploaded_file = st.sidebar.file_uploader(
    "Upload VIW_FID_EPI.csv",
    type=["csv"],
    help="File should contain date, country, and case-count columns.",
)

if uploaded_file is None:
    st.info("Upload VIW_FID_EPI.csv in the sidebar to begin.")
    st.stop()

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    # Basic sanity: try to parse date col if present
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    return df

df = load_data(uploaded_file)

#st.subheader("Raw data preview")
#st.write(df.head())

# Check required columns
missing = [c for c in [DATE_COL, COUNTRY_COL, CASES_COL] if c not in df.columns]
if missing:
    st.error(
        f"Missing expected columns: {missing}. "
        f"Please edit DATE_COL / COUNTRY_COL / CASES_COL at the top of the script "
        f"to match your CSV schema."
    )
    st.stop()

# Drop rows with invalid dates or cases
df = df.dropna(subset=[DATE_COL, COUNTRY_COL, CASES_COL])
df = df[df[CASES_COL].notna()]

# Filter to only include rows where AGEGROUP_CODE equals "All" (case-insensitive)
if "AGEGROUP_CODE" in df.columns:
    df = df[df["AGEGROUP_CODE"].str.upper() == "ALL"].copy()

# --- Controls ---
st.sidebar.header("Filters")

countries = sorted(df[COUNTRY_COL].dropna().unique())
selected_countries = st.sidebar.multiselect(
    "Countries",
    countries,
    default=countries[:1],  # or [] if you prefer starting empty
)

if not selected_countries:
    st.warning("Select at least one country to display.")
    st.stop()

country_df = df[df[COUNTRY_COL].isin(selected_countries)].copy()
if country_df.empty:
    st.warning("No data for the selected countries.")
    st.stop()

country_df = country_df.sort_values(DATE_COL)
country_df["year"] = country_df[DATE_COL].dt.year

min_year = int(country_df["year"].min())
max_year = int(country_df["year"].max())

if max_year - min_year < 2:
    st.warning(
        f"Data for selected countries spans {min_year}–{max_year} "
        f"(less than 3 full years); showing all available years."
    )
    use_window = False
else:
    use_window = True

if use_window:
    start_year = st.sidebar.slider(
        "3‑year window start year",
        min_value=min_year,
        max_value=max_year - 2,
        value=min_year,
        step=1,
    )
    end_year = start_year + 2
    window_df = country_df[
        (country_df["year"] >= start_year) & (country_df["year"] <= end_year)
    ]
else:
    window_df = country_df.copy()
    start_year, end_year = min_year, max_year

# --- Plot ---
st.subheader("Influenza cases over time")

if window_df.empty:
    st.warning("No observations in the selected 3‑year window.")
else:
    title = (
        f"{', '.join(selected_countries)}: {CASES_COL} over time "
        f"({start_year}–{end_year}, 3‑year window)"
        if use_window
        else f"{', '.join(selected_countries)}: {CASES_COL} over time (all available years)"
    )

    # Aggregate by country + week: sum cases for rows with the same country and week/date
    weekly_df = (
        window_df
        .groupby([COUNTRY_COL, DATE_COL], as_index=False)[CASES_COL]
        .sum()
    )

    fig = px.line(
        weekly_df,
        x=DATE_COL,
        y=CASES_COL,
        color=COUNTRY_COL,
        title=title,
        labels={DATE_COL: "Week start", CASES_COL: "Cases (sum per week)", COUNTRY_COL: "Country"},
    )
    fig.update_traces(mode="lines+markers")
    st.plotly_chart(fig, use_container_width=True)

# --- Summary statistics ---
st.subheader("Summary statistics (selected country & window)")

if window_df.empty:
    st.info("No data to summarize for the selected window.")
else:
    total_cases = window_df[CASES_COL].sum()
    mean_cases = window_df[CASES_COL].mean()
    max_cases = window_df[CASES_COL].max()
    observations = window_df.shape[0]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Countries", ", ".join(selected_countries))
    col2.metric("Window", f"{start_year}–{end_year}")
    col3.metric("Total cases", f"{total_cases:,.2f}")
    col4.metric("Mean cases / obs", f"{mean_cases:,.2f}")
    col5.metric("Max cases", f"{max_cases:,.2f}")

    with st.expander("Show filtered data table"):
        st.dataframe(window_df[[DATE_COL, COUNTRY_COL, CASES_COL]].reset_index(drop=True))