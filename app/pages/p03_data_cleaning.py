import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from app.utils.datasets import DATASETS
from app.utils.notebook_export import make_notebook, TOPIC_NOTEBOOKS
from app.utils.visualizations import plot_distribution, plot_missing_heatmap, fig_defaults
import plotly.express as px
import plotly.graph_objects as go

def render():
    st.markdown("""
    <div class="hero-banner" style="padding:1.5rem;">
      <h1 style="font-size:1.8rem;">🧹 Data Cleaning</h1>
      <p>Where 80% of real data science work lives — missing values, outliers, duplicates</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📖 Explain", "🎨 Interactive Cleaner", "💻 Code", "✅ Quiz"])

    with tab1:
        st.markdown("### Why Cleaning Matters")
        st.markdown("""
> **Garbage In → Garbage Out**
> A model trained on dirty data is **more dangerous** than no model — it gives confident wrong answers.

Data cleaning is the process of detecting and correcting (or removing) corrupt, inaccurate,
or irrelevant records from a dataset.
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### The 5 Types of Dirty Data")
            st.code("""
1. MISSING VALUES
   sensor_temp = [22.1, NaN, 23.4, NaN, 22.8]
   Cause: sensor dropout, network loss, human error

2. OUTLIERS / ANOMALIES
   engine_rpm = [3000, 3100, 45000, 3050]
   Cause: sensor spike, data entry error, real event

3. DUPLICATES
   Same event logged twice (network retry)

4. INCONSISTENCIES
   material = ["steel", "Steel", "STEEL", "stl"]

5. WRONG TYPES / FORMATS
   date = "21-06-2026" vs "2026/06/21" vs "Jun 21"
            """, language="text")

        with col2:
            st.markdown("### Cleaning Strategy Decision Tree")
            st.code("""
Missing value detected
       │
       ├─► < 5% missing?
       │      ├─ YES → Drop rows (safe)
       │      └─ NO  → Continue...
       │
       ├─► Numerical column?
       │      ├─ YES → Fill with median/mean/interpolate
       │      └─ NO  → Fill with mode or "Unknown"
       │
       └─► High missingness (>50%)?
              ├─ YES → Drop the column
              └─ NO  → Use ML imputation (KNNImputer)

Outlier detected
       │
       ├─► Domain knowledge: valid range?
       │      ├─ Outside → Cap or remove
       │      └─ Inside  → Keep (could be real signal!)
       │
       └─► Use IQR or Z-score method
            """, language="text")

        st.markdown("---")
        st.markdown("""
        <div class="box-info">
        <strong>🧠 AI Impact:</strong> Scikit-learn models silently fail on NaN values.
        Neural networks amplify outlier noise during gradient descent.
        A single bad row can shift a regression line by 30%.
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("### Interactive Data Cleaning Lab")

        ds_key = st.selectbox("Choose dataset:", list(DATASETS.keys()))
        df_raw = DATASETS[ds_key]()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total rows", f"{len(df_raw):,}")
        col2.metric("Missing values", df_raw.isnull().sum().sum())
        col3.metric("Duplicate rows", df_raw.duplicated().sum())

        st.markdown("#### Raw Data (with issues)")
        st.dataframe(df_raw.head(15), use_container_width=True)

        st.markdown("#### Missing Values Heatmap")
        fig_miss = plot_missing_heatmap(df_raw)
        if fig_miss:
            st.plotly_chart(fig_miss, use_container_width=True)

        st.markdown("---")
        st.markdown("### Apply Cleaning Steps")

        df_clean = df_raw.copy()

        num_cols = df_raw.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df_raw.select_dtypes(include="object").columns.tolist()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Missing Value Strategy**")
            miss_strategy = st.radio("Numerical columns:", [
                "Fill with median", "Fill with mean", "Drop rows", "Forward fill"
            ], key="miss_num")
            cat_fill = st.radio("Categorical columns:", [
                "Fill with 'Unknown'", "Fill with mode", "Drop rows"
            ], key="miss_cat")

        with col2:
            st.markdown("**Outlier Treatment**")
            outlier_method = st.radio("Method:", [
                "None (keep all)", "IQR capping (1.5×)", "Z-score removal (±3σ)", "Percentile capping (1-99%)"
            ], key="outlier")
            dup_action = st.radio("Duplicates:", ["Keep all", "Drop duplicates"], key="dups")

        if st.button("🧹 Apply Cleaning", type="primary"):
            # Missing values — numerical
            for col in num_cols:
                if df_clean[col].isnull().any():
                    if miss_strategy == "Fill with median":
                        df_clean[col].fillna(df_clean[col].median(), inplace=True)
                    elif miss_strategy == "Fill with mean":
                        df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                    elif miss_strategy == "Drop rows":
                        df_clean.dropna(subset=[col], inplace=True)
                    elif miss_strategy == "Forward fill":
                        df_clean[col].fillna(method="ffill", inplace=True)

            # Missing values — categorical
            for col in cat_cols:
                if df_clean[col].isnull().any():
                    if cat_fill == "Fill with 'Unknown'":
                        df_clean[col].fillna("Unknown", inplace=True)
                    elif cat_fill == "Fill with mode":
                        df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
                    elif cat_fill == "Drop rows":
                        df_clean.dropna(subset=[col], inplace=True)

            # Outliers
            if outlier_method == "IQR capping (1.5×)":
                for col in num_cols:
                    Q1, Q3 = df_clean[col].quantile(0.25), df_clean[col].quantile(0.75)
                    IQR = Q3 - Q1
                    df_clean[col] = df_clean[col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)
            elif outlier_method == "Z-score removal (±3σ)":
                from scipy import stats
                z_scores = np.abs(stats.zscore(df_clean[num_cols].fillna(0)))
                df_clean = df_clean[(z_scores < 3).all(axis=1)]
            elif outlier_method == "Percentile capping (1-99%)":
                for col in num_cols:
                    p1, p99 = df_clean[col].quantile(0.01), df_clean[col].quantile(0.99)
                    df_clean[col] = df_clean[col].clip(p1, p99)

            # Duplicates
            if dup_action == "Drop duplicates":
                df_clean.drop_duplicates(inplace=True)

            st.markdown("### Cleaning Results")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows before", f"{len(df_raw):,}")
            c2.metric("Rows after", f"{len(df_clean):,}", delta=f"{len(df_clean)-len(df_raw)}")
            c3.metric("Missing before", df_raw.isnull().sum().sum())
            c4.metric("Missing after", df_clean.isnull().sum().sum(),
                      delta=str(df_clean.isnull().sum().sum() - df_raw.isnull().sum().sum()))

            st.markdown("#### Cleaned Data Preview")
            st.dataframe(df_clean.head(15), use_container_width=True)

            if num_cols:
                col_compare = num_cols[0]
                fig_compare = go.Figure()
                fig_compare.add_trace(go.Box(y=df_raw[col_compare].dropna(), name="Before", marker_color="#ef4444"))
                fig_compare.add_trace(go.Box(y=df_clean[col_compare].dropna(), name="After", marker_color="#10b981"))
                fig_compare = fig_defaults(fig_compare, f"{col_compare}: Before vs After Cleaning")
                st.plotly_chart(fig_compare, use_container_width=True)

            st.download_button("⬇ Download Cleaned CSV", df_clean.to_csv(index=False),
                               file_name="cleaned_data.csv", mime="text/csv")

    with tab3:
        st.markdown("### Cleaning Code Toolkit")
        nb_bytes = make_notebook("Data Cleaning — DS Workshop 2026", TOPIC_NOTEBOOKS.get("Data Cleaning", []))
        st.download_button("📓 Download as Jupyter Notebook", nb_bytes,
                           file_name="03_data_cleaning.ipynb", mime="application/json")
        st.markdown("---")
                st.code("""
import pandas as pd
import numpy as np

df = pd.read_csv("raw_data.csv")

# ── 1. Audit ────────────────────────────────────────────────────────
print(df.shape)
print(df.isnull().sum())                    # missing per column
print(df.isnull().mean() * 100)             # missing percentage
print(df.duplicated().sum())                # duplicate rows
print(df.describe())                        # catch impossible ranges

# ── 2. Missing Values ────────────────────────────────────────────────
df["temp"].fillna(df["temp"].median(), inplace=True)     # robust fill
df["category"].fillna("Unknown", inplace=True)           # categorical
df["signal"].fillna(method="ffill", inplace=True)        # time series
df.dropna(subset=["critical_col"], inplace=True)         # must-have cols

# Advanced: scikit-learn imputer
from sklearn.impute import SimpleImputer, KNNImputer
imp = KNNImputer(n_neighbors=5)
df[numeric_cols] = imp.fit_transform(df[numeric_cols])

# ── 3. Outliers ──────────────────────────────────────────────────────
# IQR method
Q1 = df["rpm"].quantile(0.25)
Q3 = df["rpm"].quantile(0.75)
IQR = Q3 - Q1
df["rpm"] = df["rpm"].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)    # cap

# Z-score method (remove rows)
from scipy.stats import zscore
df = df[(np.abs(zscore(df[numeric_cols])) < 3).all(axis=1)]

# ── 4. Duplicates ────────────────────────────────────────────────────
df.drop_duplicates(inplace=True)
df.drop_duplicates(subset=["event_id"], keep="last", inplace=True)

# ── 5. String Cleaning ───────────────────────────────────────────────
df["material"] = df["material"].str.strip().str.title()
df["material"] = df["material"].replace({"Stl": "Steel", "Al": "Aluminum"})

# ── 6. Type Fixing ───────────────────────────────────────────────────
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df["value"] = pd.to_numeric(df["value"], errors="coerce")

# ── 7. Validation ────────────────────────────────────────────────────
assert df["temperature"].between(-50, 1000).all(), "Temperature out of range!"
assert df.isnull().sum().sum() == 0, "Still has missing values!"
        """, language="python")

    with tab4:
        st.markdown("### Quiz — Data Cleaning")
        questions = [
            {"q": "A sensor column has 3% missing values. What's the safest strategy?",
             "options": ["Drop the column", "Fill with median", "Drop rows", "Leave as NaN"],
             "answer": "Drop rows", "explanation": "3% is small — dropping rows loses minimal data and is the cleanest approach."},
            {"q": "Temperature readings show values of -999. What type of data issue is this?",
             "options": ["Missing values", "Outlier", "Sentinel/error code", "Duplicate"],
             "answer": "Sentinel/error code", "explanation": "-999 is typically a sentinel value used by sensors to indicate 'no reading'. Must be replaced with NaN."},
            {"q": "What does IQR stand for in outlier detection?",
             "options": ["Internal Query Range", "Interquartile Range", "Index Query Ratio", "Inverse Quality Ratio"],
             "answer": "Interquartile Range", "explanation": "IQR = Q3 - Q1. Values outside Q1 - 1.5×IQR or Q3 + 1.5×IQR are flagged as outliers."},
        ]
        for i, q in enumerate(questions):
            st.markdown(f"**Q{i+1}.** {q['q']}")
            choice = st.radio("", q["options"], key=f"quiz_03_q{i}", index=None, horizontal=True)
            if choice:
                if choice == q["answer"]:
                    st.markdown(f'<div class="box-success">✅ Correct! {q["explanation"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="box-warning">❌ Answer: **{q["answer"]}**. {q["explanation"]}</div>', unsafe_allow_html=True)
            st.markdown("")
        if "data_cleaning" not in st.session_state.completed_topics:
            if st.button("✅ Mark topic as complete"):
                st.session_state.completed_topics.append("data_cleaning")
                st.success("Topic marked complete!")
