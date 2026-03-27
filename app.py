import json
import pickle
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ---- Compatibility for notebook-defined classes used inside pipeline pickle ----
try:
    import ames_pipeline  # noqa: F401
except Exception:
    ames_pipeline = None

import __main__

if ames_pipeline is not None:
    __main__.AmesFeatureBuilder = ames_pipeline.AmesFeatureBuilder
    __main__.MeanTargetEncoder = ames_pipeline.MeanTargetEncoder
    __main__.LogTransformer = ames_pipeline.LogTransformer

# sklearn pickle compatibility shim
try:
    import sklearn.compose._column_transformer as _ct

    class _RemainderColsList(list):
        pass

    _ct._RemainderColsList = _RemainderColsList
except Exception:
    pass


st.set_page_config(
    page_title="Ames Housing Intelligence",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #ffffff !important;
        margin-bottom: 0;
        line-height: 1.1;
    }

    .sub-header {
        font-size: 1rem;
        color: #e5e7eb !important;
        margin-top: 0.25rem;
        font-weight: 400;
    }

    .section-title {
        font-size: 1.45rem;
        font-weight: 700;
        color: #ffffff !important;
        border-bottom: 2px solid rgba(255,255,255,0.18);
        padding-bottom: 0.35rem;
        margin-top: 1.75rem;
        margin-bottom: 0.75rem;
    }

    .helper-box {
        background: #eef6ff;
        border-left: 4px solid #3b82f6;
        border-radius: 0 10px 10px 0;
        padding: 0.9rem 1rem;
        color: #12304a;
        margin: 0.8rem 0 1rem 0;
        font-size: 0.95rem;
    }

    .insight-box {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        border-radius: 0 10px 10px 0;
        padding: 0.85rem 1rem;
        color: #78350f;
        margin: 0.75rem 0;
        font-size: 0.93rem;
    }

    .soft-box {
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 0.9rem 1rem;
        margin: 0.7rem 0;
        color: #111827;
        font-size: 0.94rem;
    }

    .price-tag {
        font-size: 3rem;
        font-weight: 800;
        color: #0f5132;
        background: #d1fae5;
        border-radius: 16px;
        padding: 0.9rem 1.6rem;
        display: inline-block;
        margin: 0.6rem 0 1rem 0;
    }

    div[data-testid="stSidebar"] {
        background-color: #1a1a2e;
    }

    div[data-testid="stSidebar"] * {
        color: #e5e7eb !important;
    }

    div[data-testid="stSidebar"] .stSelectbox label,
    div[data-testid="stSidebar"] .stSlider label,
    div[data-testid="stSidebar"] .stNumberInput label {
        color: #cbd5e1 !important;
        font-size: 0.8rem !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

REQUIRED_FILES = [
    "train.csv",
    "model.pkl",
    "defaults.json",
    "ui_feature_plan.json",
    "model_metadata.json",
]


def patch_loaded_pipeline(obj, seen=None):
    if seen is None:
        seen = set()
    oid = id(obj)
    if oid in seen:
        return
    seen.add(oid)

    if hasattr(obj, "_fit_dtype") and not hasattr(obj, "_fill_dtype"):
        try:
            obj._fill_dtype = obj._fit_dtype
        except Exception:
            pass

    for attr in ["steps", "transformers", "transformers_"]:
        if hasattr(obj, attr):
            try:
                for item in getattr(obj, attr):
                    if isinstance(item, tuple):
                        for sub in item[1:]:
                            patch_loaded_pipeline(sub, seen)
                    else:
                        patch_loaded_pipeline(item, seen)
            except Exception:
                pass

    if hasattr(obj, "named_steps"):
        try:
            for sub in obj.named_steps.values():
                patch_loaded_pipeline(sub, seen)
        except Exception:
            pass


@st.cache_resource(show_spinner=False)
def load_pipeline():
    with open("model.pkl", "rb") as f:
        loaded = pickle.load(f)
    patch_loaded_pipeline(loaded)
    return loaded


@st.cache_data(show_spinner=False)
def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv("train.csv")


missing = [f for f in REQUIRED_FILES if not Path(f).exists()]
if missing:
    st.error(f"Missing required files: {missing}")
    st.stop()

pipeline = load_pipeline()
defaults = load_json("defaults.json")
ui_plan = load_json("ui_feature_plan.json")
meta = load_json("model_metadata.json")
df_raw = load_data()

FINAL_TEST = meta.get("final_metrics", {}).get("test", {})
TOP_FEATURES = meta.get("top_transformed_features", [])
UI_FEATURES = ui_plan.get("ui_features", [])


def get_price_prediction(raw_row: dict) -> float:
    X = pd.DataFrame([raw_row])
    pred_log = float(pipeline.predict(X)[0])
    return float(np.expm1(pred_log))


QUALITY_OPTS = ["Po", "Fa", "TA", "Gd", "Ex"]
BASEMENT_OPTS = ["None", "No", "Mn", "Av", "Gd"]
FIREPLACE_OPTS = ["None", "Po", "Fa", "TA", "Gd", "Ex"]
GARAGE_FINISH_OPTS = ["None", "Unf", "RFn", "Fin"]


def num_default(col, fallback=None):
    val = defaults.get(col, fallback)
    if val is None:
        series = pd.to_numeric(df_raw[col], errors="coerce")
        return float(series.median())
    return float(val)


def section_note(text: str):
    st.markdown(f"<div class='helper-box'>{text}</div>", unsafe_allow_html=True)


def soft_note(text: str):
    st.markdown(f"<div class='soft-box'>{text}</div>", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("## 🏠 Valora")
    st.markdown('Your digital compass for real estate value.')
    st.markdown("---")

    page = st.radio(
        "Navigate",
        [
            "Overview",
            "What-If Simulator",
            "Neighbourhood Analysis",
            "Model Explainability",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**Final model snapshot**")
    st.metric("Test R²", f"{FINAL_TEST.get('R2', 0):.3f}")
    st.metric("Test RMSE (log)", f"{FINAL_TEST.get('RMSE', 0):.3f}")
    st.metric("Test MAE (log)", f"{FINAL_TEST.get('MAE', 0):.3f}")
    st.caption("These are holdout test metrics from the final notebook pipeline.")

    st.markdown("---")
    st.markdown("**Data at a glance**")
    st.caption(f"Rows: {len(df_raw):,}")
    st.caption(f"Train rows: {meta.get('train_rows', '—')}")
    st.caption(f"Test rows: {meta.get('test_rows', '—')}")
    st.caption(f"Interactive UI inputs: {len(UI_FEATURES)}")

st.markdown('<p class="main-header">Ames Housing Price Intelligence</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Explore the housing market, test house profiles, and predict property values.</p>',
    unsafe_allow_html=True,
)
st.markdown("---")

if page == "Overview":


    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Median sale price", f"${df_raw['SalePrice'].median():,.0f}", delta="Ames, Iowa")
    with c2:
        st.metric("Price range", f"${df_raw['SalePrice'].min():,.0f} – ${df_raw['SalePrice'].max():,.0f}")
    with c3:
        st.metric("Test R²", f"{FINAL_TEST.get('R2', 0):.3f}", delta=f"RMSE(log) {FINAL_TEST.get('RMSE', 0):.3f}")
    with c4:
        st.metric("Total properties", f"{len(df_raw):,}", delta=f"{df_raw['YrSold'].nunique()} years of sales")

    st.markdown('<p class="section-title">Sale price distribution</p>', unsafe_allow_html=True)
    

    col_a, col_b = st.columns([3, 2])
    with col_a:
        fig = px.histogram(df_raw, x="SalePrice", nbins=60, template="plotly_white")
        fig.update_traces(opacity=0.85)
        fig.add_vline(
            x=df_raw["SalePrice"].median(),
            line_dash="dash",
            line_color="#D85A30",
            annotation_text="Median",
            annotation_position="top right",
        )
        fig.update_layout(height=310, margin=dict(t=10, b=10, l=10, r=10), xaxis_tickprefix="$", xaxis_tickformat=",.0f")
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        yr_med = df_raw.groupby("YrSold", as_index=False)["SalePrice"].median()
        fig2 = px.line(yr_med, x="YrSold", y="SalePrice", markers=True, template="plotly_white")
        fig2.update_layout(height=310, margin=dict(t=10, b=10, l=10, r=10), yaxis_tickprefix="$", yaxis_tickformat=",.0f")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<p class="section-title">Key feature relationships</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(df_raw, x="GrLivArea", y="SalePrice", color="OverallQual", opacity=0.6, template="plotly_white")
        fig.update_layout(height=360, margin=dict(t=10, b=10, l=10, r=10), yaxis_tickprefix="$", yaxis_tickformat=",.0f")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        qual_med = df_raw.groupby("OverallQual", as_index=False)["SalePrice"].median()
        fig = px.bar(qual_med, x="OverallQual", y="SalePrice", color="SalePrice", template="plotly_white")
        fig.update_layout(height=360, margin=dict(t=10, b=10, l=10, r=10), yaxis_tickprefix="$", yaxis_tickformat=",.0f", coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="section-title">Market drift check</p>', unsafe_allow_html=True)
    year_line = df_raw.groupby("YrSold", as_index=False)["SalePrice"].median()
    fig = px.line(year_line, x="YrSold", y="SalePrice", markers=True, template="plotly_white")
    fig.update_layout(height=320, margin=dict(t=10, b=10, l=10, r=10), yaxis_tickprefix="$", yaxis_tickformat=",.0f")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="section-title">Feature comparison: earlier vs later sales</p>', unsafe_allow_html=True)
    feature_to_check = st.selectbox("Choose a feature to compare", ["OverallQual", "GrLivArea", "YearBuilt", "TotalBsmtSF", "GarageArea"])
    early = df_raw.loc[df_raw["YrSold"] <= 2008, feature_to_check]
    late = df_raw.loc[df_raw["YrSold"] >= 2009, feature_to_check]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=early, name="Earlier sales (2006–2008)", opacity=0.65, nbinsx=30))
    fig.add_trace(go.Histogram(x=late, name="Later sales (2009–2010)", opacity=0.65, nbinsx=30))
    fig.update_layout(barmode="overlay", height=320, margin=dict(t=10, b=10, l=10, r=10), template="plotly_white", legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, use_container_width=True)

elif page == "What-If Simulator":
    st.markdown('<p class="section-title">What-If Price Simulator</p>', unsafe_allow_html=True)


    raw_row = defaults.copy()
    left, right = st.columns([1.05, 0.95])

    with left:
        st.markdown("**Step 1: choose the main house details**")
        neighborhoods = sorted(df_raw["Neighborhood"].dropna().unique().tolist())
        raw_row["Neighborhood"] = st.selectbox(
            "Neighbourhood",
            neighborhoods,
            index=neighborhoods.index(defaults.get("Neighborhood", neighborhoods[0])) if defaults.get("Neighborhood") in neighborhoods else 0,
            help="People often compare prices by location first, so neighbourhood is included as a practical user-facing input.",
        )
        raw_row["OverallQual"] = st.slider("Overall quality", 1, 10, int(round(num_default("OverallQual", 6))), step=1)
        raw_row["GrLivArea"] = st.slider("Above-ground living area (sq ft)", 400, 4500, int(round(num_default("GrLivArea", 1500))), step=25)
        raw_row["TotalBsmtSF"] = st.slider("Basement area (sq ft)", 0, 3500, int(round(num_default("TotalBsmtSF", 900))), step=25)
        raw_row["1stFlrSF"] = st.slider("1st floor area (sq ft)", 300, 3000, int(round(num_default("1stFlrSF", 1000))), step=25)
        raw_row["2ndFlrSF"] = st.slider("2nd floor area (sq ft)", 0, 2500, int(round(num_default("2ndFlrSF", 0))), step=25)
        raw_row["YearBuilt"] = st.slider("Year built", 1870, 2010, int(round(num_default("YearBuilt", 1970))), step=1)
        raw_row["YearRemodAdd"] = st.slider("Year remodeled", 1950, 2010, int(round(num_default("YearRemodAdd", 1990))), step=1)
        raw_row["KitchenQual"] = st.selectbox("Kitchen quality", QUALITY_OPTS, index=QUALITY_OPTS.index(defaults.get("KitchenQual", "TA")) if defaults.get("KitchenQual", "TA") in QUALITY_OPTS else 2)
        raw_row["FullBath"] = st.slider("Full bathrooms", 0, 4, int(round(num_default("FullBath", 2))), step=1)
        raw_row["GarageCars"] = st.slider("Garage capacity (cars)", 0, 4, int(round(num_default("GarageCars", 2))), step=1)

        with st.expander("Step 2: optional advanced controls"):
            st.caption("Use these only if you want a more detailed house profile. Otherwise the app keeps reasonable default values.")
            raw_row["OverallCond"] = st.slider("Overall condition", 1, 10, int(round(num_default("OverallCond", 5))), step=1)
            raw_row["LotArea"] = st.slider("Lot area (sq ft)", 1000, 100000, int(round(num_default("LotArea", 9500))), step=250)
            raw_row["BsmtFinSF1"] = st.slider("Finished basement area (sq ft)", 0, 2500, int(round(num_default("BsmtFinSF1", 350))), step=25)
            raw_row["GarageArea"] = st.slider("Garage area (sq ft)", 0, 1500, int(round(num_default("GarageArea", 450))), step=10)
            raw_row["BedroomAbvGr"] = st.slider("Bedrooms above grade", 0, 8, int(round(num_default("BedroomAbvGr", 3))), step=1)
            raw_row["TotRmsAbvGrd"] = st.slider("Total rooms above grade", 2, 15, int(round(num_default("TotRmsAbvGrd", 6))), step=1)
            raw_row["Fireplaces"] = st.slider("Number of fireplaces", 0, 4, int(round(num_default("Fireplaces", 1))), step=1)
            raw_row["FireplaceQu"] = st.selectbox("Fireplace quality", FIREPLACE_OPTS, index=FIREPLACE_OPTS.index(defaults.get("FireplaceQu", "Gd")) if defaults.get("FireplaceQu", "Gd") in FIREPLACE_OPTS else 0)
            raw_row["GarageFinish"] = st.selectbox("Garage finish", GARAGE_FINISH_OPTS, index=GARAGE_FINISH_OPTS.index(defaults.get("GarageFinish", "Unf")) if defaults.get("GarageFinish", "Unf") in GARAGE_FINISH_OPTS else 1)
            raw_row["BsmtExposure"] = st.selectbox("Basement exposure", BASEMENT_OPTS, index=BASEMENT_OPTS.index(defaults.get("BsmtExposure", "No")) if defaults.get("BsmtExposure", "No") in BASEMENT_OPTS else 1)

        current_pred = get_price_prediction(raw_row)
        baseline_pred = get_price_prediction(defaults.copy())
        delta_pct = ((current_pred - baseline_pred) / baseline_pred * 100.0) if baseline_pred else 0.0


    with right:
        st.markdown("**Step 3: read the estimated result**")
        st.markdown(f'<div class="price-tag">${current_pred:,.0f}</div>', unsafe_allow_html=True)
        st.metric("Change vs default house profile", f"{delta_pct:+.1f}%")
        st.metric("Estimated price per sq ft", f"${current_pred / max(float(raw_row.get('GrLivArea', 1)), 1):,.0f}")
        st.metric("Selected neighbourhood", raw_row.get("Neighborhood", ""))
        st.metric("Year built", f"{int(raw_row.get('YearBuilt', 0))}")



        nbhd = raw_row.get("Neighborhood")
        nbhd_median = df_raw.loc[df_raw["Neighborhood"] == nbhd, "SalePrice"].median()
        if pd.notna(nbhd_median) and nbhd_median > 0:
            diff_pct = (current_pred - nbhd_median) / nbhd_median * 100
            st.markdown(f"**Compared with the neighbourhood median** (${nbhd_median:,.0f})")
            st.markdown(f"<div class='soft-box'>The estimate is <b>{diff_pct:+.1f}%</b> relative to the median observed sale price for this neighbourhood.</div>", unsafe_allow_html=True)

        changed = [c for c in UI_FEATURES if raw_row.get(c) != defaults.get(c)]
        if changed:
            st.markdown("**Inputs changed from the default profile**")
            st.write(", ".join(changed))
        else:
            st.caption("You are currently viewing the default house profile used by the app.")

    st.markdown('<p class="section-title">Sensitivity check: quality vs predicted price</p>', unsafe_allow_html=True)
    qual_range = list(range(1, 11))
    qual_preds = []
    for q in qual_range:
        row_q = raw_row.copy()
        row_q["OverallQual"] = q
        qual_preds.append(get_price_prediction(row_q))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=qual_range, y=qual_preds, mode="lines+markers", name="Predicted price"))
    fig.add_vline(x=int(raw_row["OverallQual"]), line_dash="dash", line_color="#D85A30")
    fig.update_layout(height=320, template="plotly_white", margin=dict(t=10, b=10, l=10, r=10), xaxis_title="Overall quality", yaxis_title="Predicted price ($)", yaxis_tickprefix="$", yaxis_tickformat=",.0f")
    st.plotly_chart(fig, use_container_width=True)

elif page == "Neighbourhood Analysis":
    st.markdown('<p class="section-title">Neighbourhood Analysis</p>', unsafe_allow_html=True)
    section_note(
        "This page is for to have an initial location-based view. Helps to have a glance how house hold price varies with location."
    )

    nbhd_stats = df_raw.groupby("Neighborhood").agg(
        median_price=("SalePrice", "median"),
        mean_price=("SalePrice", "mean"),
        min_price=("SalePrice", "min"),
        max_price=("SalePrice", "max"),
        count=("SalePrice", "count"),
        avg_qual=("OverallQual", "mean"),
        avg_sqft=("GrLivArea", "mean"),
    ).reset_index().sort_values("median_price", ascending=False)
    nbhd_stats["price_per_sqft"] = nbhd_stats["mean_price"] / nbhd_stats["avg_sqft"].replace(0, np.nan)

    soft_note(
        "The bar chart ranks neighbourhoods by median sale price. The small summary panel on the right helps to inspect one selected neighbourhood in more detail."
    )
    col1, col2 = st.columns([3, 2])
    with col1:
        fig = px.bar(nbhd_stats, x="median_price", y="Neighborhood", orientation="h", color="avg_qual", template="plotly_white")
        fig.update_layout(height=600, margin=dict(t=10, b=10, l=10, r=10), xaxis_tickprefix="$", xaxis_tickformat=",.0f")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        selected = st.selectbox("Choose a neighbourhood", nbhd_stats["Neighborhood"].tolist())
        row = nbhd_stats.loc[nbhd_stats["Neighborhood"] == selected].iloc[0]
        st.metric("Median price", f"${row['median_price']:,.0f}")
        st.metric("Average price per sq ft", f"${row['price_per_sqft']:.0f}")
        st.metric("Average overall quality", f"{row['avg_qual']:.1f} / 10")
        st.metric("Average living area", f"{row['avg_sqft']:,.0f} sq ft")
        st.metric("Recorded sales", f"{int(row['count'])}")
        fig2 = px.histogram(df_raw.loc[df_raw["Neighborhood"] == selected, "SalePrice"], nbins=20, template="plotly_white")
        fig2.update_layout(height=220, margin=dict(t=10, b=10, l=10, r=10), xaxis_tickprefix="$", xaxis_tickformat=",.0f", showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<p class="section-title">Neighbourhood positioning map</p>', unsafe_allow_html=True)
    soft_note(
        "Each bubble is one neighbourhood. Higher points are more expensive, farther right means higher average quality, larger bubbles mean more sales, and color reflects average price per square foot."
    )
    fig = px.scatter(nbhd_stats, x="avg_qual", y="median_price", size="count", color="price_per_sqft", hover_name="Neighborhood", template="plotly_white")
    fig.update_layout(height=420, margin=dict(t=10, b=10, l=10, r=10), yaxis_tickprefix="$", yaxis_tickformat=",.0f")
    st.plotly_chart(fig, use_container_width=True)

elif page == "Model Explainability":
    st.markdown('<p class="section-title">Model Explainability</p>', unsafe_allow_html=True)
    section_note(
        "This page shows which transformed model features had the strongest average influence while prediction and also used to understand the model's behavior."
    )
    soft_note(
        "These are mean absolute SHAP values. In simple words, larger bars mean the model relied more on that feature on average when making predictions."
    )

    imp_df = pd.DataFrame(TOP_FEATURES).copy()
    if len(imp_df.columns) >= 2:
        imp_df.columns = ["Feature", "Mean |SHAP|"]

    col1, col2 = st.columns([3, 2])
    with col1:
        fig = px.bar(imp_df.head(20), x="Mean |SHAP|", y="Feature", orientation="h", color="Mean |SHAP|", template="plotly_white")
        fig.update_layout(height=520, margin=dict(t=10, b=10, l=10, r=10), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("**Top average drivers**")
        max_imp = imp_df["Mean |SHAP|"].max() if not imp_df.empty else 0
        for i, row in imp_df.head(8).iterrows():
            width = int((row["Mean |SHAP|"] / max_imp) * 100) if max_imp else 0
            st.markdown(
                f"**{i+1}. {row['Feature']}**  \n"
                f"<div style='background:#e8eaf0;border-radius:4px;height:6px;width:100%'>"
                f"<div style='background:#1D9E75;border-radius:4px;height:6px;width:{width}%'></div></div>",
                unsafe_allow_html=True,
            )
            st.caption(f"Mean |SHAP|: {row['Mean |SHAP|']:.4f}")


