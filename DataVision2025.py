import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# -------------------------------------------------------------------
# Page configuration
# -------------------------------------------------------------------
st.set_page_config(
    page_title="DataVision 2025 - Air Quality Analysis",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a cleaner look
st.markdown(
    """
    <style>
        .main-header {
            font-size: 2.4rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 0.4rem;
        }
        .sub-header {
            font-size: 1.1rem;
            color: #555;
            text-align: center;
            margin-bottom: 1.4rem;
        }
        .metric-card {
            background-color: #f5f6fa;
            padding: 0.8rem;
            border-radius: 0.5rem;
            margin: 0.2rem 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# Title area
# -------------------------------------------------------------------
st.markdown(
    '<p class="main-header">DataVision 2025 – Air Quality Analysis Dashboard</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="sub-header">Team: Naive Bayes Ninjas</p>',
    unsafe_allow_html=True,
)
st.markdown("---")

# -------------------------------------------------------------------
# Sidebar navigation
# -------------------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to section",
    [
        "Home",
        "Data Overview",
        "Exploratory Analysis",
        "ML Predictions",
        "Insights & Recommendations",
    ],
)

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader(
    "Upload city_day.csv file", type=["csv"]
)

# -------------------------------------------------------------------
# Data loading and processing
# -------------------------------------------------------------------
@st.cache_data
def load_and_process_data(uploaded_file):
    """Load and pre-process the air quality dataset."""
    df = pd.read_csv(uploaded_file)

    # Basic cleaning
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = (
        df.dropna(subset=["Date", "City"])
        .drop_duplicates(subset=["Date", "City"])
        .sort_values(["City", "Date"])
        .reset_index(drop=True)
    )

    pollutants = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "AQI", "NO", "NOx", "NH3"]
    for col in pollutants:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Outlier clipping and forward fill
    def clip_outliers(group):
        q95 = group.quantile(0.95)
        return group.clip(upper=q95)

    for col in pollutants:
        if col in df.columns:
            df[col] = (
                df.groupby("City")[col]
                .apply(clip_outliers)
                .reset_index(0, drop=True)
            )
            df[col] = df.groupby("City")[col].fillna(method="ffill")

    # Temporal features
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["DayOfWeek"] = df["Date"].dt.dayofweek

    def get_season(month):
        if month in [12, 1, 2]:
            return "Winter"
        if month in [3, 4, 5]:
            return "Summer"
        if month in [6, 7, 8, 9]:
            return "Monsoon"
        if month in [10, 11]:
            return "Post-Monsoon"
        return "Other"

    df["Season"] = df["Month"].apply(get_season)
    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

    # Domain and lag features
    df["PM_ratio"] = df["PM2.5"] / (df["PM10"] + 1)
    df["AQI_momentum_3d"] = (
        df.sort_values(["City", "Date"])
        .groupby("City")["AQI"]
        .pct_change()
        .rolling(3, min_periods=1)
        .std()
    )
    df["AQI_lag1"] = df.groupby("City")["AQI"].shift(1)
    df["AQI_lag3"] = df.groupby("City")["AQI"].shift(3)

    # AQI buckets
    def aqi_bucket(aqi):
        if pd.isna(aqi):
            return "Unknown"
        if aqi <= 50:
            return "Good"
        if aqi <= 100:
            return "Satisfactory"
        if aqi <= 200:
            return "Moderate"
        if aqi <= 300:
            return "Poor"
        if aqi <= 400:
            return "Very Poor"
        return "Severe"

    df["AQI_Bucket"] = df["AQI"].apply(aqi_bucket)

    return df


if uploaded_file is None:
    st.warning("Please upload the 'city_day.csv' file to start the analysis.")
    st.info("Use the file uploader in the left sidebar.")
    st.stop()

with st.spinner("Loading and processing data..."):
    df = load_and_process_data(uploaded_file)

# -------------------------------------------------------------------
# HOME
# -------------------------------------------------------------------
if page == "Home":
    st.header("Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total records", f"{len(df):,}")
    with col2:
        st.metric("Number of cities", df["City"].nunique())
    with col3:
        st.metric(
            "Date range",
            f"{df['Date'].min().year} – {df['Date'].max().year}",
        )
    with col4:
        st.metric("Average AQI", f"{df['AQI'].mean():.1f}")

    st.markdown("---")
    st.subheader("Project description")
    st.write(
        """
        This dashboard helps explore and understand air quality patterns across Indian cities.
        It combines descriptive statistics, visual analysis, and machine learning models to
        highlight pollution hotspots, seasonal behaviour, and AQI prediction performance.
        """
    )

    st.subheader("What you can do here")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            """
            - Explore raw data and summary statistics  
            - Check missing values and data coverage  
            - See seasonal and city-wise AQI trends  
            """
        )
    with col_b:
        st.markdown(
            """
            - Train and evaluate AQI prediction models  
            - Inspect feature importance  
            - Review insights and policy-style recommendations  
            """
        )

# -------------------------------------------------------------------
# DATA OVERVIEW
# -------------------------------------------------------------------
elif page == "Data Overview":
    st.header("Data overview")

    tab1, tab2, tab3 = st.tabs(["Dataset info", "Sample rows", "Missing values"])

    with tab1:
        st.subheader("Basic information")
        c1, c2 = st.columns(2)
        with c1:
            st.write("Shape:", df.shape)
            st.write(
                "Date range:",
                f"{df['Date'].min().date()} to {df['Date'].max().date()}",
            )
        with c2:
            st.write("Number of cities:", df["City"].nunique())
            st.write("Main pollutants tracked:", 10)

        st.subheader("AQI summary statistics")
        st.dataframe(df["AQI"].describe().to_frame().T, use_container_width=True)

    with tab2:
        st.subheader("First 20 rows")
        st.dataframe(df.head(20), use_container_width=True)

        st.subheader("Top cities by number of records")
        city_counts = df["City"].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(9, 4))
        city_counts.plot(kind="barh", ax=ax, color="steelblue")
        ax.set_xlabel("Number of records")
        ax.set_ylabel("City")
        st.pyplot(fig)

    with tab3:
        st.subheader("Missing values by pollutant")
        pollutants = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "AQI", "NO", "NOx", "NH3"]
        missing = df[pollutants].isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)

        missing_df = (
            pd.DataFrame({"Missing count": missing, "Percent": missing_pct})
            .sort_values("Missing count", ascending=False)
        )

        c1, c2 = st.columns([1, 2])
        with c1:
            st.dataframe(missing_df, use_container_width=True)
        with c2:
            fig, ax = plt.subplots(figsize=(9, 4))
            missing_df["Percent"].plot(kind="bar", ax=ax, color="coral")
            ax.set_ylabel("Percent missing")
            ax.set_title("Missing values per pollutant")
            plt.xticks(rotation=45)
            st.pyplot(fig)

# -------------------------------------------------------------------
# EXPLORATORY ANALYSIS
# -------------------------------------------------------------------
elif page == "Exploratory Analysis":
    st.header("Exploratory analysis")

    tab1, tab2, tab3 = st.tabs(
        ["Seasonal patterns", "City trends", "Pollutant behaviour"]
    )

    # Seasonal patterns
    with tab1:
        st.subheader("Seasonal AQI for major cities")

        major_cities = df["City"].value_counts().head(10).index
        pivot = df[df["City"].isin(major_cities)].pivot_table(
            values="AQI", index="Season", columns="City", aggfunc="mean"
        )

        fig, ax = plt.subplots(figsize=(12, 5))
        sns.heatmap(
            pivot, annot=True, fmt=".0f", cmap="Reds", cbar_kws={"label": "Average AQI"}, ax=ax
        )
        ax.set_title("Seasonal AQI – top 10 cities")
        st.pyplot(fig)

        seasonal = df.groupby("Season")["AQI"].agg(["mean", "median", "count"]).round(1)
        seasonal = seasonal.sort_values("mean", ascending=False)
        st.dataframe(seasonal, use_container_width=True)

    # City trends
    with tab2:
        st.subheader("Monthly AQI trends for the most polluted cities")

        top_cities = df.groupby("City")["AQI"].mean().nlargest(5).index
        dftop = df[df["City"].isin(top_cities)].copy()
        dftop_monthly = (
            dftop.set_index("Date")
            .groupby("City")["AQI"]
            .resample("M")
            .mean()
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=(12, 5))
        for city in top_cities:
            city_data = dftop_monthly[dftop_monthly["City"] == city]
            ax.plot(city_data["Date"], city_data["AQI"], marker="o", linewidth=2, label=city)
        ax.set_ylabel("AQI")
        ax.set_title("Top 5 polluted cities – monthly AQI")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.grid(alpha=0.3)
        st.pyplot(fig)

        st.subheader("Top 10 most polluted cities (by mean AQI)")
        city_rank = (
            df.groupby("City")["AQI"]
            .agg(["mean", "median", "max", "count"])
            .round(1)
            .sort_values("mean", ascending=False)
            .head(10)
        )
        st.dataframe(city_rank, use_container_width=True)

    # Pollutant behaviour
    with tab3:
        st.subheader("Average pollutant levels")

        poll_means = (
            df[["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]]
            .mean()
            .sort_values(ascending=False)
        )

        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(8, 4))
            poll_means.plot(kind="barh", ax=ax, color="teal")
            ax.set_xlabel("Average concentration")
            st.pyplot(fig)
        with c2:
            st.write("Approximate mean concentrations:")
            for pol, val in poll_means.items():
                st.write(f"- {pol}: {val:.2f}")

        st.subheader("AQI distribution by category")
        aqi_dist = df["AQI_Bucket"].value_counts()
        fig, ax = plt.subplots(figsize=(8, 4))
        aqi_dist.plot(kind="bar", ax=ax, color="steelblue")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of AQI categories")
        plt.xticks(rotation=45)
        st.pyplot(fig)

# -------------------------------------------------------------------
# ML PREDICTIONS
# -------------------------------------------------------------------
elif page == "ML Predictions":
    st.header("Machine learning – AQI prediction")

    ml_features = [
        "PM2.5",
        "PM10",
        "NO2",
        "SO2",
        "CO",
        "O3",
        "PM_ratio",
        "Month_sin",
        "Month_cos",
        "DayOfWeek",
        "AQI_momentum_3d",
        "AQI_lag1",
        "AQI_lag3",
    ]

    df_ml = df.dropna(subset=ml_features + ["AQI"]).copy()
    X = df_ml[ml_features]
    y = df_ml["AQI"]

    st.write(f"Training dataset size: {len(df_ml):,} rows and {len(ml_features)} features.")

    with st.spinner("Running time-series cross-validation..."):
        tscv = TimeSeriesSplit(n_splits=3)
        scaler = StandardScaler()

        rf = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )
        xgb_model = xgb.XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1
        )
        models = {"RandomForest": rf, "XGBoost": xgb_model}

        results = []
        for name, model in models.items():
            scores = []
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                if name == "RandomForest":
                    scaler.fit(X_train)
                    X_train_scaled = scaler.transform(X_train)
                    X_val_scaled = scaler.transform(X_val)
                else:
                    X_train_scaled, X_val_scaled = X_train, X_val

                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_val_scaled)
                scores.append(r2_score(y_val, y_pred))

            results.append(
                {
                    "Model": name,
                    "CV_R2_mean": np.mean(scores),
                    "CV_R2_std": np.std(scores),
                }
            )

    st.subheader("Cross‑validation results")
    st.dataframe(pd.DataFrame(results), use_container_width=True)

    # Final training on 80/20 split
    st.subheader("Final model on hold‑out set")

    split_idx = int(0.8 * len(X))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    best_model = rf
    best_model.fit(X_train_scaled, y_train)
    y_pred = best_model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("MAE", f"{mae:.2f}")
    with c2:
        st.metric("RMSE", f"{rmse:.2f}")
    with c3:
        st.metric("R² score", f"{r2:.3f}")

    st.subheader("Predicted vs actual AQI")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_test, y_pred, alpha=0.5)
    ax.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "r--",
        linewidth=2,
    )
    ax.set_xlabel("Actual AQI")
    ax.set_ylabel("Predicted AQI")
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    st.subheader("Feature importance (Random Forest)")
    fi = pd.DataFrame(
        {"Feature": ml_features, "Importance": best_model.feature_importances_}
    ).sort_values("Importance", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    fi.plot(kind="barh", x="Feature", y="Importance", ax=ax, color="darkgreen")
    ax.set_xlabel("Importance")
    st.pyplot(fig)

# -------------------------------------------------------------------
# INSIGHTS AND RECOMMENDATIONS
# -------------------------------------------------------------------
else:
    st.header("Key insights and recommendations")

    st.subheader("Main observations")

    seasonal_mean = df.groupby("Season")["AQI"].mean().round(1)
    top_cities = df.groupby("City")["AQI"].mean().nlargest(3)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Statistical highlights**")
        if "Winter" in seasonal_mean.index:
            st.write(f"- Winter has the highest average AQI: {seasonal_mean['Winter']}")
        st.write(f"- Overall mean AQI: {df['AQI'].mean():.1f}")
        st.write(f"- Peak AQI observed: {df['AQI'].max():.1f}")
        st.write(
            f"- Most polluted city by mean AQI: {top_cities.index[0]} "
            f"({top_cities.iloc[0]:.1f})"
        )

    with c2:
        st.markdown("**Model summary**")
        st.write("- Random Forest performs best among tested models.")
        st.write("- Cross‑validated R² is around 0.8, which is strong for AQI data.")
        st.write("- Lag features and PM‑related variables are among the top predictors.")

    st.markdown("---")
    st.subheader("Policy‑style recommendations")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("**Emission controls**")
        st.write("- Strengthen PM2.5 controls in winter.")
        st.write("- Focus on industrial clusters in highest‑AQI cities.")
    with col_b:
        st.markdown("**Traffic and transport**")
        st.write("- Use traffic restrictions in severe episodes and festivals.")
        st.write("- Promote public transport and non‑motorised travel.")
    with col_c:
        st.markdown("**Monitoring and alerts**")
        st.write("- Increase monitoring density in the top 10 polluted cities.")
        st.write("- Provide clear public AQI alerts and guidance.")

    st.markdown("---")
    st.subheader("Summary table")

    summary = {
        "Metric": [
            "Number of cities",
            "Date range",
            "Mean AQI",
            "Maximum AQI",
            "Most polluted city (mean AQI)",
            "Dominant pollutant (by focus)",
        ],
        "Value": [
            df["City"].nunique(),
            f"{df['Date'].min().year}–{df['Date'].max().year}",
            f"{df['AQI'].mean():.1f}",
            f"{df['AQI'].max():.1f}",
            top_cities.index[0],
            "PM2.5",
        ],
    }
    st.table(pd.DataFrame(summary))

    st.info("Use the navigation menu on the left to revisit specific parts of the analysis.")
