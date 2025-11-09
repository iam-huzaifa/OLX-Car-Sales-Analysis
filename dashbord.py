import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------- Page Setup -------------------
st.set_page_config(page_title="OLX Used Cars Dashboard", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    .main {background-color: #f9f9f9;}
    h1, h2, h3, h4 {color: #1a1a1a;}
    .stMetric {background: white; padding: 12px; border-radius: 12px; box-shadow: 0 2px 6px rgba(0,0,0,0.1);}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🚗 OLX Used Cars — Interactive Dashboard")
st.markdown("Analyze used car listings uploaded from OLX or similar marketplaces. Upload your dataset or use a local CSV file.")

# ------------------- Sidebar Section -------------------
st.sidebar.header("📂 Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

# Safe CSV loader
def load_csv(file):
    try:
        return pd.read_csv(file, encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(file, encoding='latin1')
    except Exception as e:
        st.error(f"❌ Could not read file: {e}")
        return None

# Load data
if uploaded_file is not None:
    df = load_csv(uploaded_file)
else:
    try:
        df = load_csv("olx_cars.csv")
        st.sidebar.info("Loaded default 'olx_cars.csv' from app folder.")
    except FileNotFoundError:
        st.sidebar.warning("No 'olx_cars.csv' found. Please upload a CSV file.")
        df = None

# ------------------- Main Dashboard -------------------
if df is not None:
    st.sidebar.success(f"✅ Loaded dataset with {df.shape[0]:,} rows and {df.shape[1]} columns")

    cols = df.columns.tolist()

    # --- Column detection helper ---
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

    st.sidebar.header("🧭 Column Mapping")
    col_price = st.sidebar.selectbox("Price column", cols, index=cols.index(find_col(["price","amount","selling_price","ad_price"])) if find_col(["price","amount","selling_price","ad_price"]) else 0)
    col_brand = st.sidebar.selectbox("Brand column", cols, index=cols.index(find_col(["brand","make","manufacturer"])) if find_col(["brand","make","manufacturer"]) else 0)
    col_model = st.sidebar.selectbox("Model column", cols, index=cols.index(find_col(["model","car_model"])) if find_col(["model","car_model"]) else 0)
    col_year = st.sidebar.selectbox("Year column", cols, index=cols.index(find_col(["year","manufacture_year","model_year"])) if find_col(["year","manufacture_year","model_year"]) else 0)
    col_mileage = st.sidebar.selectbox("Mileage column", cols, index=cols.index(find_col(["mileage","km","kilometers","kms"])) if find_col(["mileage","km","kilometers","kms"]) else 0)
    col_city = st.sidebar.selectbox("City column", cols, index=cols.index(find_col(["city","location","town"])) if find_col(["city","location","town"]) else 0)

    # --- Data Cleaning ---
    df_filtered = df.copy()
    if col_year:
        df_filtered[col_year] = pd.to_numeric(df_filtered[col_year], errors="coerce")
    if col_price:
        df_filtered[col_price] = pd.to_numeric(df_filtered[col_price].astype(str).str.replace("[^0-9.]","",regex=True), errors="coerce")

    # --- Sidebar Filters ---
    st.sidebar.header("🔍 Filters")

    if col_brand:
        brands = ["All"] + sorted(df_filtered[col_brand].dropna().astype(str).unique().tolist())
        sel_brand = st.sidebar.selectbox("Brand", brands, index=0)
        if sel_brand != "All":
            df_filtered = df_filtered[df_filtered[col_brand].astype(str) == sel_brand]

    if col_year:
        yr_min, yr_max = int(df_filtered[col_year].min(skipna=True)), int(df_filtered[col_year].max(skipna=True))
        sel_year = st.sidebar.slider("Year range", yr_min, yr_max, (yr_min, yr_max))
        df_filtered = df_filtered[(df_filtered[col_year] >= sel_year[0]) & (df_filtered[col_year] <= sel_year[1])]

    if col_price:
        pmin, pmax = int(df_filtered[col_price].min(skipna=True)), int(df_filtered[col_price].max(skipna=True))
        sel_price = st.sidebar.slider("Price range", pmin, pmax, (pmin, pmax))
        df_filtered = df_filtered[(df_filtered[col_price] >= sel_price[0]) & (df_filtered[col_price] <= sel_price[1])]

    # ------------------- Metrics -------------------
    st.markdown("### 📊 Key Metrics")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Ads", f"{df_filtered.shape[0]:,}")
    with c2:
        st.metric("Avg Price", f"Rs {df_filtered[col_price].mean():,.0f}" if col_price else "N/A")
    with c3:
        st.metric("Avg Mileage", f"{df_filtered[col_mileage].astype(float).mean():,.0f}" if col_mileage else "N/A")
    with c4:
        st.metric("Cities", f"{df_filtered[col_city].nunique() if col_city else 0}")

    st.markdown("---")

    # ------------------- Charts -------------------
    st.markdown("### 📈 Visual Insights")
    left, right = st.columns((2,1))

    with left:
        if col_price and col_year:
            fig = px.box(df_filtered, x=col_year, y=col_price, points="outliers", title="Price by Year", color_discrete_sequence=["#2E86C1"])
            st.plotly_chart(fig, use_container_width=True)
        elif col_price:
            fig = px.histogram(df_filtered, x=col_price, nbins=40, title="Price Distribution", color_discrete_sequence=["#3498DB"])
            st.plotly_chart(fig, use_container_width=True)

    with right:
        if col_brand:
            top_brands = df_filtered[col_brand].value_counts().nlargest(10)
            fig2 = px.bar(x=top_brands.index, y=top_brands.values, labels={'x':col_brand,'y':'Count'}, title="Top 10 Brands", color_discrete_sequence=["#1ABC9C"])
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔍 Additional Analysis")
    a, b = st.columns(2)
    with a:
        if col_mileage and col_price:
            fig3 = px.scatter(df_filtered, x=col_mileage, y=col_price, trendline="ols", title="Price vs Mileage", opacity=0.6, color_discrete_sequence=["#E74C3C"])
            st.plotly_chart(fig3, use_container_width=True)
    with b:
        if col_city and col_price:
            city_price = df_filtered.groupby(col_city)[col_price].median().sort_values(ascending=False).head(15)
            fig4 = px.bar(x=city_price.index, y=city_price.values, labels={'x':col_city,'y':'Median Price'}, title="Top Cities by Median Price", color_discrete_sequence=["#9B59B6"])
            st.plotly_chart(fig4, use_container_width=True)

    # ------------------- Data Table -------------------
    st.markdown("---")
    st.markdown("### 📋 Filtered Data Preview")
    st.dataframe(df_filtered.reset_index(drop=True), use_container_width=True)

    # Download filtered data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df(df_filtered)
    st.download_button("💾 Download Filtered Data", data=csv, file_name="filtered_olx_data.csv", mime="text/csv")

else:
    st.info("👈 Upload a CSV file or place 'olx_cars.csv' in your folder to get started.")
