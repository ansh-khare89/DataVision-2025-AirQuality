import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
from datetime import date

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="DataVision 2025 - Air Quality Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* {font-family: 'Inter', sans-serif;}
.main-header {font-size: 3.2rem; font-weight: 700; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center;}
.sub-header {font-size: 1.4rem; color: #64748b; text-align: center; font-weight: 400;}
.metric-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1.8rem; border-radius: 20px; box-shadow: 0 12px 35px rgba(102,126,234,0.3); margin: 0.5rem 0; transition: all 0.3s;}
.metric-card:hover {transform: translateY(-8px); box-shadow: 0 20px 45px rgba(102,126,234,0.4);}
.stTabs [data-baseweb="tab"] {height: 55px; padding: 0 28px; background: #f8fafc; border-radius: 12px; font-weight: 600; border: 2px solid #e2e8f0;}
div[data-testid="stMetricValue"] {font-size: 2.2rem; font-weight: 700;}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────
st.markdown('<p class="main-header">DataVision 2025</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced Air Quality Intelligence Platform — Team: Naive Bayes Ninjas</p>', unsafe_allow_html=True)
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.title("Control Center")
st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader(
    "Upload Dataset (optional)",
    type="csv",
    help="Upload your own CSV. If left empty, the built-in dataset.csv loads automatically."
)

st.sidebar.subheader("Controls")
forecast_days = st.sidebar.slider("Forecast Days", 7, 30, 14)

# ── Data processing ───────────────────────────────────────────
@st.cache_data
def process_data(source):
    df = pd.read_csv(source)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "City", "AQI"])
    df = df.sort_values(["City", "Date"]).reset_index(drop=True)

    num_cols = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "AQI"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].ffill().bfill()

    df["Month"]     = df["Date"].dt.month
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

    if "PM2.5" in df.columns and "PM10" in df.columns:
        df["PM_ratio"] = df["PM2.5"] / (df["PM10"] + 1)

    def aqi_category(aqi):
        if aqi <= 50:  return "Good"
        if aqi <= 100: return "Satisfactory"
        if aqi <= 200: return "Moderate"
        if aqi <= 300: return "Poor"
        if aqi <= 400: return "Very Poor"
        return "Severe"

    df["AQI_Category"] = df["AQI"].apply(aqi_category)
    return df

# ── Load: uploaded file → dataset.csv → friendly error ───────
if uploaded_file is not None:
    source = uploaded_file
    st.sidebar.success("✅ Using uploaded file.")
else:
    source = "dataset.csv"
    st.sidebar.info("ℹ️ Using built-in dataset.csv")

try:
    with st.spinner("Loading and processing data..."):
        df = process_data(source)
except FileNotFoundError:
    st.error(
        "**dataset.csv not found.**\n\n"
        "Please ensure `dataset.csv` is in the same folder as `app.py`, "
        "or upload your CSV file using the sidebar."
    )
    st.info("Expected columns: Date, City, AQI, PM2.5, PM10, NO2, SO2, CO, O3")
    st.stop()

st.success(f"Dataset loaded: **{len(df):,} records** | **{df['City'].nunique()} cities**")

# ── Sidebar filters (after data loads) ───────────────────────
city_filter = st.sidebar.multiselect(
    "Filter Cities",
    options=sorted(df["City"].unique()),
    default=sorted(df["City"].unique())[:5]
)
date_range = st.sidebar.date_input("Date Range", value=(date(2015, 1, 1), date.today()))

# ── Apply filters ─────────────────────────────────────────────
df_f = df.copy()
if city_filter:
    df_f = df_f[df_f["City"].isin(city_filter)]
if len(date_range) == 2:
    df_f = df_f[
        (df_f["Date"] >= pd.to_datetime(date_range[0])) &
        (df_f["Date"] <= pd.to_datetime(date_range[1]))
    ]

# ── KPI row ───────────────────────────────────────────────────
st.header("Executive Dashboard")
c1, c2, c3, c4, c5 = st.columns(5)
for col, label, value in zip(
    [c1, c2, c3, c4, c5],
    ["Total Records", "Cities", "Average AQI", "Peak AQI", "Severe Days"],
    [f"{len(df_f):,}", str(df_f["City"].nunique()),
     f"{df_f['AQI'].mean():.1f}", f"{df_f['AQI'].max():.0f}",
     f"{(df_f['AQI'] > 300).sum():,}"]
):
    col.markdown(f"""
    <div class="metric-card">
        <div style="font-size:0.85rem;opacity:0.9;">{label}</div>
        <div style="font-size:2.2rem;font-weight:700;">{value}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ── Charts row 1 ─────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("AQI Trends Over Time")
    top6 = df_f.groupby("City")["AQI"].mean().nlargest(6).index
    trend = df_f[df_f["City"].isin(top6)].groupby(["Date","City"])["AQI"].mean().reset_index()
    fig = px.line(trend, x="Date", y="AQI", color="City", title="Top 6 Cities — AQI Evolution")
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("AQI Category Distribution")
    cat = df_f["AQI_Category"].value_counts()
    fig = px.pie(values=cat.values, names=cat.index, title="Air Quality Categories",
                 color_discrete_sequence=["#10b981","#f59e0b","#f97316","#ef4444","#dc2626","#b91c1c"])
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

# ── Charts row 2 ─────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Top Polluted Cities")
    city_aqi = df_f.groupby("City")["AQI"].mean().nlargest(10).sort_values().reset_index()
    fig = px.bar(city_aqi, x="AQI", y="City", orientation="h",
                 color="AQI", color_continuous_scale="Reds",
                 title="Top 10 Most Polluted Cities (Avg AQI)")
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Pollutant Correlations")
    corr_cols = [c for c in ["PM2.5","PM10","NO2","SO2","CO","O3"] if c in df_f.columns]
    if len(corr_cols) > 1:
        fig = px.imshow(df_f[corr_cols].corr().round(2), text_auto=True,
                        color_continuous_scale="RdBu_r", title="Pollutant Correlation Matrix")
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 AQI Predictor", "🤖 ML Pipeline", "📈 Forecasting"])

# Tab 1 ── Live predictor ─────────────────────────────────────
with tab1:
    st.subheader("Live AQI Prediction Engine")
    ca, cb, cc = st.columns(3)
    with ca: pm25  = st.slider("PM2.5 (µg/m³)", 0.0, 500.0, 150.0)
    with cb: pm10  = st.slider("PM10 (µg/m³)",  0.0, 1000.0, 250.0)
    with cc: no2   = st.slider("NO2 (µg/m³)",   0.0, 200.0,  50.0)
    month = st.slider("Month", 1, 12, 12,
                      format="%d", help="1=Jan … 12=Dec")

    if st.button("Generate AQI Prediction", type="primary", use_container_width=True):
        pred = 0.8*pm25 + 0.3*pm10 + 0.5*no2 + (20 if month in [12,1,2] else 0)
        st.metric("Predicted AQI", f"{pred:.0f}")
        if pred > 300:   st.error("🔴 Severe — Immediate action required")
        elif pred > 200: st.warning("🟠 Poor — Health advisory in effect")
        elif pred > 100: st.warning("🟡 Moderate air quality")
        else:            st.success("🟢 Good / Satisfactory air quality")

# Tab 2 ── ML pipeline ────────────────────────────────────────
with tab2:
    st.subheader("Machine Learning Pipeline")
    feats  = [c for c in ["PM2.5","PM10","NO2","SO2","CO","O3","month_sin","month_cos"] if c in df.columns]
    df_ml  = df[feats + ["AQI"]].dropna()

    if len(df_ml) < 100:
        st.warning("Not enough data to train the model.")
    else:
        with st.spinner("Training Random Forest…"):
            X = df_ml[feats]; y = df_ml["AQI"]
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
            mdl = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
            mdl.fit(X_tr, y_tr)
            preds = mdl.predict(X_te)

        r2   = r2_score(y_te, preds)
        mae  = mean_absolute_error(y_te, preds)
        rmse = np.sqrt(mean_squared_error(y_te, preds))

        m1, m2, m3 = st.columns(3)
        m1.metric("R² Score", f"{r2:.3f}")
        m2.metric("MAE",      f"{mae:.1f}")
        m3.metric("RMSE",     f"{rmse:.1f}")

        imp = pd.DataFrame({"Feature": feats, "Importance": mdl.feature_importances_}).sort_values("Importance")
        st.plotly_chart(px.bar(imp, x="Importance", y="Feature", orientation="h",
                               title="Feature Importance"), use_container_width=True)

        fig2 = px.scatter(x=y_te, y=preds, opacity=0.4,
                          labels={"x":"Actual AQI","y":"Predicted AQI"},
                          title=f"Predicted vs Actual (R²={r2:.3f})")
        fig2.add_shape(type="line", x0=float(y_te.min()), y0=float(y_te.min()),
                       x1=float(y_te.max()), y1=float(y_te.max()),
                       line=dict(color="red", dash="dash"))
        st.plotly_chart(fig2, use_container_width=True)

# Tab 3 ── Forecasting ────────────────────────────────────────
with tab3:
    st.subheader(f"AQI Baseline Forecast — Next {forecast_days} Days")
    dates    = pd.date_range(start=df["Date"].max() + pd.Timedelta(days=1),
                             periods=forecast_days, freq="D")
    baseline = df["AQI"].mean()
    forecast = baseline + np.random.normal(0, df["AQI"].std() * 0.1, forecast_days)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=forecast, mode="lines+markers",
                             name="Forecast", line=dict(color="#667eea", width=2)))
    fig.add_hline(y=200, line_dash="dash", line_color="orange",
                  annotation_text="Poor threshold (200)")
    fig.add_hline(y=300, line_dash="dash", line_color="red",
                  annotation_text="Severe threshold (300)")
    fig.update_layout(title=f"AQI Forecast — Next {forecast_days} Days",
                      xaxis_title="Date", yaxis_title="AQI", height=500)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("*DataVision 2025 | Air Quality Intelligence Platform | Team: Naive Bayes Ninjas*")
