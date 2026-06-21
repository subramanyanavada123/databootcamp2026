import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from app.utils.datasets import DATASETS
from app.utils.notebook_export import make_notebook, TOPIC_NOTEBOOKS
from app.utils.visualizations import plot_feature_importance, fig_defaults
import plotly.express as px

def render():
    st.markdown("""
    <div class="hero-banner" style="padding:1.5rem;">
      <h1 style="font-size:1.8rem;">🛠️ Feature Engineering</h1>
      <p>The art of creating signal — turning domain knowledge into model inputs</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📖 Explain", "🎨 Feature Lab", "💻 Code", "✅ Quiz"])

    with tab1:
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown("### What is Feature Engineering?")
            st.markdown("""
Feature engineering is creating new input variables (**features**) from existing raw data
that better represent the underlying problem to the ML model.

> "Applied ML is basically feature engineering." — Andrew Ng

Raw data rarely tells the model what to pay attention to.
You — the engineer with domain knowledge — must *teach* the model through the features you create.
            """)
            st.code("""
RAW DATA                  ENGINEERED FEATURES
────────────────────────────────────────────────────────────
timestamp              →  hour, day_of_week, is_weekend
start_time, end_time   →  duration_minutes
lat, lon               →  distance_from_hub, zone_cluster
failure = True         →  days_since_last_failure
rpm, torque            →  power = rpm × torque / 9549
price, cost            →  profit_margin = (price-cost)/price
text review            →  sentiment_score, word_count
image pixel            →  edge_density, brightness, HOG
            """, language="text")

        with col2:
            st.markdown("### Feature Engineering Techniques")
            techniques = {
                "Technique": ["Interaction terms", "Polynomial", "Binning", "Aggregation", "Ratio features", "Lag features", "Rolling stats", "Target encoding"],
                "Example": ["A × B", "x²", "age → Young/Mid/Senior", "mean per group", "price/cost", "value at t-1", "7-day average", "mean(target) per category"],
                "Model type": ["Linear", "Linear", "Tree", "Any", "Any", "Time series", "Time series", "Tree/Neural"],
            }
            st.dataframe(pd.DataFrame(techniques), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("""
        <div class="box-info">
        <strong>🧠 AI Perspective:</strong>
        Deep learning can auto-engineer features from raw images/text.
        But for tabular data, manual feature engineering still beats AutoML in most competitions.
        The winning Kaggle solution is usually 20% better algorithm, 80% better features.
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("### Feature Engineering Lab")
        ds_key = st.selectbox("Choose dataset:", list(DATASETS.keys()))
        df = DATASETS[ds_key]()
        st.markdown("#### Original Dataset")
        st.dataframe(df.head(5), use_container_width=True)

        st.markdown("### Create New Features")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        date_cols = df.select_dtypes(include="datetime").columns.tolist()

        df_eng = df.copy()
        new_features = []

        exp1 = st.expander("📐 Interaction Features (A × B or A / B)", expanded=True)
        with exp1:
            if len(num_cols) >= 2:
                c1, c2, c3 = st.columns(3)
                col_a = c1.selectbox("Column A:", num_cols, key="fe_a")
                op    = c2.selectbox("Operation:", ["×  Multiply", "÷  Divide", "+  Add", "−  Subtract"])
                col_b = c3.selectbox("Column B:", [c for c in num_cols if c != col_a], key="fe_b")
                if st.button("Create Interaction Feature"):
                    new_name = f"{col_a}_x_{col_b}" if "×" in op else f"{col_a}_div_{col_b}" if "÷" in op else f"{col_a}_plus_{col_b}" if "+" in op else f"{col_a}_minus_{col_b}"
                    if "×" in op:
                        df_eng[new_name] = df_eng[col_a] * df_eng[col_b]
                    elif "÷" in op:
                        df_eng[new_name] = df_eng[col_a] / (df_eng[col_b].replace(0, np.nan))
                    elif "+" in op:
                        df_eng[new_name] = df_eng[col_a] + df_eng[col_b]
                    else:
                        df_eng[new_name] = df_eng[col_a] - df_eng[col_b]
                    new_features.append(new_name)
                    st.success(f"Created feature: **{new_name}**")

        exp2 = st.expander("📅 Datetime Features")
        with exp2:
            if date_cols:
                dt_col = st.selectbox("Date column:", date_cols)
                opts = st.multiselect("Extract:", ["hour", "day", "month", "year", "dayofweek", "is_weekend", "quarter"])
                if st.button("Extract Date Features") and opts:
                    for o in opts:
                        if o == "is_weekend":
                            df_eng["is_weekend"] = df_eng[dt_col].dt.dayofweek.isin([5, 6]).astype(int)
                            new_features.append("is_weekend")
                        else:
                            df_eng[o] = getattr(df_eng[dt_col].dt, o)
                            new_features.append(o)
                    st.success(f"Created: {opts}")
            else:
                st.info("No datetime columns in this dataset.")

        exp3 = st.expander("📦 Binning (Numerical → Category)")
        with exp3:
            if num_cols:
                bin_col = st.selectbox("Column to bin:", num_cols, key="fe_bin")
                n_bins  = st.slider("Number of bins:", 2, 10, 3)
                bin_labels = st.text_input("Labels (comma-separated, optional):", placeholder="Low,Medium,High")
                if st.button("Create Bins"):
                    labels = [l.strip() for l in bin_labels.split(",")] if bin_labels else None
                    if labels and len(labels) != n_bins:
                        st.warning(f"Need exactly {n_bins} labels.")
                    else:
                        new_name = f"{bin_col}_bin"
                        df_eng[new_name] = pd.cut(df_eng[bin_col], bins=n_bins, labels=labels)
                        new_features.append(new_name)
                        st.success(f"Created: **{new_name}**")

        exp4 = st.expander("📊 Group Aggregations (mean/std per category)")
        with exp4:
            if cat_cols and num_cols:
                c1, c2, c3 = st.columns(3)
                grp_col = c1.selectbox("Group by:", cat_cols, key="fe_grp")
                agg_col = c2.selectbox("Aggregate:", num_cols, key="fe_agg")
                agg_fn  = c3.selectbox("Function:", ["mean", "std", "max", "min", "count"])
                if st.button("Create Group Feature"):
                    new_name = f"{grp_col}_{agg_fn}_{agg_col}"
                    grp_map = df_eng.groupby(grp_col)[agg_col].transform(agg_fn)
                    df_eng[new_name] = grp_map
                    new_features.append(new_name)
                    st.success(f"Created: **{new_name}**")

        if new_features:
            st.markdown("### New Features Created")
            st.write(", ".join([f"**{f}**" for f in new_features]))
            st.dataframe(df_eng[new_features].head(10), use_container_width=True)

            st.markdown("### Feature Importance (Quick RF)")
            bool_cols = df_eng.select_dtypes(include="bool").columns.tolist()
            target_candidates = bool_cols + ["passed_qc", "alert_active", "churned", "readmitted", "motion_detected"]
            target = next((c for c in target_candidates if c in df_eng.columns), None)
            if target:
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.impute import SimpleImputer
                X_cols = [c for c in df_eng.select_dtypes(include=np.number).columns if c != target]
                if X_cols:
                    X = df_eng[X_cols].copy()
                    y = df_eng[target].astype(int)
                    imp = SimpleImputer(strategy="median")
                    X_imp = imp.fit_transform(X)
                    rf = RandomForestClassifier(n_estimators=50, random_state=42)
                    rf.fit(X_imp, y)
                    fig_fi = plot_feature_importance(X_cols, rf.feature_importances_.tolist())
                    st.plotly_chart(fig_fi, use_container_width=True)

    with tab3:
        st.markdown("### Feature Engineering Code")
        nb_bytes = make_notebook("ML Fundamentals — DS Workshop 2026", TOPIC_NOTEBOOKS.get("ML Fundamentals", []))
        st.download_button("📓 Download as Jupyter Notebook", nb_bytes,
                           file_name="06_feature_engineering.ipynb", mime="application/json")
        st.markdown("---")
                st.code("""
import pandas as pd
import numpy as np

df = pd.read_parquet("clean_data.parquet")

# ── 1. Interaction features ────────────────────────────────────────────
df["power_kw"]         = df["rpm"] * df["torque"] / 9549
df["profit_margin"]    = (df["revenue"] - df["cost"]) / df["revenue"]
df["temp_deviation"]   = df["temperature"] - df["temperature"].mean()

# ── 2. Datetime decomposition ──────────────────────────────────────────
df["hour"]        = df["timestamp"].dt.hour
df["dayofweek"]   = df["timestamp"].dt.dayofweek
df["month"]       = df["timestamp"].dt.month
df["is_weekend"]  = df["dayofweek"].isin([5, 6]).astype(int)
df["is_business_hours"] = df["hour"].between(9, 17).astype(int)

# ── 3. Rolling / lag features (time series) ────────────────────────────
df = df.sort_values("timestamp")
df["temp_lag_1"]    = df["temperature"].shift(1)       # previous reading
df["temp_roll_5m"]  = df["temperature"].rolling(5).mean()  # 5-point average
df["temp_roll_std"] = df["temperature"].rolling(5).std()   # volatility
df["temp_diff"]     = df["temperature"].diff()             # rate of change

# ── 4. Group aggregations (statistical features) ──────────────────────
df["mean_temp_per_line"]  = df.groupby("line_id")["temperature"].transform("mean")
df["std_temp_per_line"]   = df.groupby("line_id")["temperature"].transform("std")
df["rank_within_line"]    = df.groupby("line_id")["temperature"].rank(pct=True)

# ── 5. Binning ────────────────────────────────────────────────────────
df["age_group"] = pd.cut(df["age"],
    bins=[0, 25, 40, 60, 100],
    labels=["Young", "Mid", "Senior", "Elder"])

df["pressure_quartile"] = pd.qcut(df["pressure"], q=4,
    labels=["Q1", "Q2", "Q3", "Q4"])

# ── 6. Text features ──────────────────────────────────────────────────
df["notes_length"]    = df["notes"].str.len()
df["has_warning"]     = df["notes"].str.contains("warn|alert|error", case=False).astype(int)
df["exclamation_cnt"] = df["notes"].str.count("!")

# ── 7. Domain-specific (manufacturing) ────────────────────────────────
# Coefficient of Variation (relative variability)
df["cv_temp"] = df["temp_roll_std"] / df["temp_roll_5m"].replace(0, np.nan)

# Process capability index (Cpk)
LSL, USL = 9.5, 10.5    # spec limits
sigma = df["thickness"].std()
mean  = df["thickness"].mean()
df["cpk"] = min((USL - mean), (mean - LSL)) / (3 * sigma)
        """, language="python")

    with tab4:
        st.markdown("### Quiz — Feature Engineering")
        questions = [
            {"q": "You have 'start_time' and 'end_time'. What feature would you create?",
             "options": ["Drop both columns", "duration = end_time - start_time", "Encode both as integers", "Bin into morning/afternoon"],
             "answer": "duration = end_time - start_time",
             "explanation": "The duration is far more informative to models than raw timestamps. Interaction features like this often have the highest importance."},
            {"q": "What is a lag feature in time series data?",
             "options": ["Log transform of the feature", "The value from a previous time step", "Feature importance rank", "Lagged mean across groups"],
             "answer": "The value from a previous time step",
             "explanation": "Lag features (value at t-1, t-2...) allow models to 'see' recent history, crucial for forecasting and anomaly detection."},
            {"q": "Which technique helps a linear model capture curved/nonlinear relationships?",
             "options": ["Label encoding", "StandardScaler", "Polynomial features (x²)", "Dropping outliers"],
             "answer": "Polynomial features (x²)",
             "explanation": "Adding x² allows linear models to fit parabolic curves. sklearn's PolynomialFeatures automates this."},
        ]
        for i, q in enumerate(questions):
            st.markdown(f"**Q{i+1}.** {q['q']}")
            choice = st.radio("", q["options"], key=f"quiz_06_q{i}", index=None, horizontal=True)
            if choice:
                if choice == q["answer"]:
                    st.markdown(f'<div class="box-success">✅ Correct! {q["explanation"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="box-warning">❌ Answer: **{q["answer"]}**. {q["explanation"]}</div>', unsafe_allow_html=True)
            st.markdown("")
        if "feature_engineering" not in st.session_state.completed_topics:
            if st.button("✅ Mark topic as complete"):
                st.session_state.completed_topics.append("feature_engineering")
                st.success("Topic marked complete!")
