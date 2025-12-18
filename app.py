import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Renewable Energy Potential Dashboard",
    layout="wide"
)

# -------------------------------------------------
# Title & Description
# -------------------------------------------------
st.title("🌱 Renewable Energy Potential Dashboard")

st.markdown("""
This dashboard presents insights from a renewable energy data analysis project.
It integrates **data exploration, feature engineering, machine learning results,**
and **visual analytics** into an interactive interface.
""")

# -------------------------------------------------
# File Upload
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "📂 Upload the cleaned renewable energy dataset (CSV)",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload `cleaned_renewable_data.csv` to continue.")
    st.stop()

df = pd.read_csv(uploaded_file)

# -------------------------------------------------
# Sidebar Filter
# -------------------------------------------------
# -------------------------------------------------
# Sidebar Filters (Multi-State Selection)
# -------------------------------------------------
st.sidebar.header("🔎 Filters")

if 'state' in df.columns:
    all_states = sorted(df['state'].dropna().unique().tolist())

    selected_states = st.sidebar.multiselect(
        "Select State(s)",
        options=all_states,
        default=all_states  # all selected by default
    )

    # Filter only if user selects something
    if selected_states:
        df = df[df['state'].isin(selected_states)]


# -------------------------------------------------
# Dataset Overview
# -------------------------------------------------
st.subheader("📋 Dataset Overview")

c1, c2, c3 = st.columns(3)
c1.metric("Rows", df.shape[0])
c2.metric("Columns", df.shape[1])
c3.metric("Numeric Features", df.select_dtypes(include='number').shape[1])

st.dataframe(df.head())

# -------------------------------------------------
# ML Model Performance Summary
# -------------------------------------------------
st.subheader("🤖 Machine Learning Model Performance")

# 🔴 UPDATE THESE VALUES FROM YOUR ML NOTEBOOK
model_scores = {
    "Linear Regression": 0.81,
    "Random Forest": 0.82
}


cols = st.columns(len(model_scores))
best_score = max(model_scores.values())
best_model = max(model_scores, key=model_scores.get)

for col, (model, score) in zip(cols, model_scores.items()):
    col.metric(
        label=model,
        value=f"{score * 100:.1f}%",
        delta="Best Model" if score == best_score else ""
    )

st.success(f"✅ Best Performing Model: **{best_model}**")

# -------------------------------------------------
# Summary Statistics
# -------------------------------------------------
st.subheader("📊 Summary Statistics")
st.dataframe(df.select_dtypes(include='number').describe())

# -------------------------------------------------
# Distribution Plot (Interactive)
# -------------------------------------------------
st.subheader("📈 Distribution Analysis")

numeric_cols = df.select_dtypes(include='number').columns.tolist()
selected_col = st.selectbox("Select a numeric column to visualize", numeric_cols)

fig, ax = plt.subplots()
ax.hist(df[selected_col].dropna(), bins=30, edgecolor="black")
ax.set_title(f"Distribution of {selected_col}")
ax.set_xlabel(selected_col)
ax.set_ylabel("Frequency")
st.pyplot(fig)

# -------------------------------------------------
# Wind Speed Comparison
# -------------------------------------------------
st.subheader("🌬️ Wind Speed Comparison")

wind_cols = [
    c for c in ['wind_speed_50m', 'wind_speed_100m', 'wind_speed_150m']
    if c in df.columns
]

if len(wind_cols) >= 2:
    fig, ax = plt.subplots()
    df[wind_cols].mean().plot(kind='bar', ax=ax)
    ax.set_ylabel("Average Wind Speed (m/s)")
    ax.set_title("Average Wind Speed at Different Heights")
    st.pyplot(fig)

# -------------------------------------------------
# Solar Energy Comparison
# -------------------------------------------------
st.subheader("☀️ Solar Energy Comparison")

solar_cols = [
    c for c in ['annual_dni_value', 'annual_ghi_value', 'annual_tilt_value']
    if c in df.columns
]

if solar_cols:
    fig, ax = plt.subplots()
    df[solar_cols].mean().plot(kind='bar', ax=ax)
    ax.set_ylabel("Average Solar Radiation (kWh/m²/day)")
    ax.set_title("Average Solar Radiation Metrics")
    st.pyplot(fig)

# -------------------------------------------------
# Renewable Energy Potential
# -------------------------------------------------
if 'renewable_score' in df.columns:
    st.subheader("⚡ Renewable Energy Potential")

    fig, ax = plt.subplots()
    ax.hist(df['renewable_score'].dropna(), bins=30, edgecolor='black')
    ax.set_title("Distribution of Renewable Energy Score")
    ax.set_xlabel("Renewable Score (0–100)")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    st.subheader("🏆 Top 10 Locations by Renewable Potential")
    display_cols = [c for c in ['state', 'city', 'renewable_score'] if c in df.columns]
    st.dataframe(
        df.sort_values('renewable_score', ascending=False)
          .head(10)[display_cols]
    )

# -------------------------------------------------
# Correlation Heatmap (Meaningful Features Only)
# -------------------------------------------------
st.subheader("🔗 Correlation Analysis")

meaningful_features = [
    'avg_wind_speed',
    'wind_speed_50m',
    'wind_speed_100m',
    'wind_speed_150m',
    'annual_dni_value',
    'annual_ghi_value',
    'annual_tilt_value',
    'solar_potential_index',
    'wind_speed_increase_rate',
    'renewable_score'
]

heatmap_cols = [c for c in meaningful_features if c in df.columns]

corr_matrix = df[heatmap_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr_matrix, cmap='coolwarm')
ax.set_xticks(range(len(heatmap_cols)))
ax.set_yticks(range(len(heatmap_cols)))
ax.set_xticklabels(heatmap_cols, rotation=90)
ax.set_yticklabels(heatmap_cols)
plt.colorbar(im)
ax.set_title("Correlation Heatmap (Physically Meaningful Features)")
st.pyplot(fig)

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.success("Dashboard loaded successfully 🎉")
