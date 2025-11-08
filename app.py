import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="OLX / Used Cars Dashboard", layout="wide", initial_sidebar_state="expanded")
st.title("🚗 OLX Used Cars — Interactive Dashboard")

# Sidebar - file upload or sample file
st.sidebar.header("Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV file (or leave empty to load local 'olx_cars.csv')", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    try:
        df = pd.read_csv("olx_cars.csv")
    except Exception:
        st.sidebar.info("No 'olx_cars.csv' found in app folder. Please upload a CSV file.")
        df = None

if df is not None:
    st.sidebar.success(f"Loaded dataset with {df.shape[0]:,} rows and {df.shape[1]} columns")
    # Show columns and let user pick important ones if autodetect fails
    st.sidebar.header("Column mapping (auto-detected)")
    cols = df.columns.tolist()

    # Try common column name variants
    def find_col(candidates):
        for c in candidates:
            for col in cols:
                if col.lower() == c.lower():
                    return col
        for c in candidates:
            for col in cols:
                if c.lower() in col.lower():
                    return col
        return None

    col_price = find_col(["price","amount","selling_price","ad_price"])
    col_brand = find_col(["make","brand","manufacturer","company"])
    col_model = find_col(["model","car_model"])
    col_year = find_col(["year","manufacture_year","model_year"])
    col_mileage = find_col(["mileage","km","kilometers","kms"])
    col_city = find_col(["city","location","town"])
    col_fuel = find_col(["fuel","fuel_type"])
    col_trans = find_col(["transmission","trans","gear"])
    col_lat = find_col(["latitude","lat"])
    col_lon = find_col(["longitude","lon","lng"])

    # Let user override mappings
    col_price = st.sidebar.selectbox("Price column", [None] + cols, index=cols.index(col_price) if col_price in cols else 0)
    col_brand = st.sidebar.selectbox("Brand column", [None] + cols, index=cols.index(col_brand) if col_brand in cols else 0)
    col_model = st.sidebar.selectbox("Model column", [None] + cols, index=cols.index(col_model) if col_model in cols else 0)
    col_year = st.sidebar.selectbox("Year column", [None] + cols, index=cols.index(col_year) if col_year in cols else 0)
    col_mileage = st.sidebar.selectbox("Mileage column", [None] + cols, index=cols.index(col_mileage) if col_mileage in cols else 0)
    col_city = st.sidebar.selectbox("City column", [None] + cols, index=cols.index(col_city) if col_city in cols else 0)

    # Basic cleaning helpers
    st.sidebar.header("Filters")
    # Create a filtered copy for interactive widgets
    df_filtered = df.copy()

    # Convert year and price to numeric if possible
    if col_year:
        df_filtered[col_year] = pd.to_numeric(df_filtered[col_year], errors="coerce")
    if col_price:
        df_filtered[col_price] = pd.to_numeric(df_filtered[col_price].astype(str).str.replace("[^0-9.]","",regex=True), errors="coerce")

    # Apply sidebar filters
    if col_brand and col_brand in df_filtered.columns:
        brands = ["All"] + sorted(df_filtered[col_brand].dropna().astype(str).unique().tolist())
        sel_brand = st.sidebar.selectbox("Brand", brands, index=0)
        if sel_brand != "All":
            df_filtered = df_filtered[df_filtered[col_brand].astype(str) == sel_brand]

    if col_year and col_year in df_filtered.columns:
        yr_min = int(df_filtered[col_year].min(skipna=True)) if not np.isnan(df_filtered[col_year].min(skipna=True)) else 2000
        yr_max = int(df_filtered[col_year].max(skipna=True)) if not np.isnan(df_filtered[col_year].max(skipna=True)) else 2025
        sel_year = st.sidebar.slider("Year range", yr_min, yr_max, (yr_min, yr_max))
        df_filtered = df_filtered[(df_filtered[col_year] >= sel_year[0]) & (df_filtered[col_year] <= sel_year[1])]

    if col_price and col_price in df_filtered.columns:
        pmin = int(df_filtered[col_price].min(skipna=True)) if not np.isnan(df_filtered[col_price].min(skipna=True)) else 0
        pmax = int(df_filtered[col_price].max(skipna=True)) if not np.isnan(df_filtered[col_price].max(skipna=True)) else 100000
        sel_price = st.sidebar.slider("Price range", pmin, pmax, (pmin, pmax))
        df_filtered = df_filtered[(df_filtered[col_price] >= sel_price[0]) & (df_filtered[col_price] <= sel_price[1])]

    # Main layout
    st.markdown("### Key Metrics")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Ads", f"{df_filtered.shape[0]:,}")
    with c2:
        if col_price:
            st.metric("Avg Price", f"Rs {df_filtered[col_price].mean():,.0f}")
        else:
            st.metric("Avg Price", "N/A")
    with c3:
        if col_mileage:
            st.metric("Avg Mileage", f"{df_filtered[col_mileage].astype(float).mean():,.0f}")
        else:
            st.metric("Avg Mileage", "N/A")
    with c4:
        st.metric("Unique Cities", f"{df_filtered[col_city].nunique() if col_city and col_city in df_filtered.columns else 0}")

    st.markdown("---")
    # Charts row 1
    st.markdown("### Charts")
    left, right = st.columns((2,1))

    with left:
        if col_price and col_year:
            fig = px.box(df_filtered, x=col_year, y=col_price, points="outliers", title="Price by Year")
            st.plotly_chart(fig, use_container_width=True)
        elif col_price:
            fig = px.histogram(df_filtered, x=col_price, nbins=30, title="Price Distribution")
            st.plotly_chart(fig, use_container_width=True)

    with right:
        if col_brand:
            top_brands = df_filtered[col_brand].value_counts().nlargest(10)
            fig2 = px.bar(x=top_brands.index, y=top_brands.values, labels={'x':col_brand,'y':'Count'}, title="Top Brands")
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    # Charts row 2
    st.markdown("### More Analysis")
    a, b = st.columns(2)
    with a:
        if col_mileage and col_price:
            fig3 = px.scatter(df_filtered, x=col_mileage, y=col_price, trendline="ols", title="Price vs Mileage")
            st.plotly_chart(fig3, use_container_width=True)
    with b:
        if col_city and col_price:
            city_price = df_filtered.groupby(col_city)[col_price].median().sort_values(ascending=False).head(15)
            fig4 = px.bar(x=city_price.index, y=city_price.values, labels={'x':col_city,'y':'Median Price'}, title="Top Cities by Median Price")
            st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")
    st.markdown("### Data Table (filtered)")
    st.dataframe(df_filtered.reset_index(drop=True), use_container_width=True)

    # Option to download filtered csv
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df_to_csv(df_filtered)
    st.download_button("Download filtered data as CSV", data=csv, file_name="olx_filtered.csv", mime="text/csv")

else:
    st.write("Upload a CSV file (or place 'olx_cars.csv' in the folder) to get started.")
