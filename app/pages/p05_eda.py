import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from app.utils.datasets import DATASETS
from app.utils.notebook_export import make_notebook, TOPIC_NOTEBOOKS
from app.utils.visualizations import (
    plot_distribution, plot_scatter, plot_correlation_heatmap,
    plot_box, plot_bar, fig_defaults
)
import plotly.express as px
import plotly.graph_objects as go

def render():
    st.markdown("""
    <div class="hero-banner" style="padding:1.5rem;">
      <h1 style="font-size:1.8rem;">🔍 Exploratory Data Analysis</h1>
      <p>Asking the right questions of your data before modeling</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📖 Explain", "🎨 EDA Workbench", "💻 Code", "✅ Quiz"])

    with tab1:
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown("### What is EDA?")
            st.markdown("""
EDA is the **detective phase** of data science. Before building any model,
you need to deeply understand your data through:

- **Univariate analysis** — one variable at a time
- **Bivariate analysis** — relationships between pairs
- **Multivariate analysis** — complex interactions
- **Temporal analysis** — trends over time

> Skipping EDA is like building a bridge without surveying the land.
            """)
            st.code("""
EDA QUESTIONS TO ANSWER
────────────────────────────────────────────────────────
 Shape:       How many rows? Columns? Data types?
 Missing:     Which columns have gaps? How many?
 Distribution: Normal? Skewed? Bimodal? Outliers?
 Correlation: Which features move together?
 Target:      What drives the outcome I care about?
 Anomalies:   Any impossible or suspicious values?
 Trends:      Does this change over time/category?
            """, language="text")

        with col2:
            st.markdown("### EDA Toolkit")
            toolkit = {
                "Analysis": ["Shape & types", "Missing values", "Distribution", "Correlation", "Outliers", "Category counts", "Time trends"],
                "Tool": ["df.info()", "df.isnull()", "df.hist()", "df.corr()", "sns.boxplot", "value_counts()", "resample()"],
            }
            st.dataframe(pd.DataFrame(toolkit), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("""
        <div class="box-info">
        <strong>🧠 AI Perspective:</strong>
        EDA reveals what features matter, guides feature engineering, and exposes leakage.
        A 3-hour EDA session can save 3 weeks of failed model iterations.
        Data scientists spend ~40% of project time in EDA.
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("### Full EDA Workbench")
        ds_key = st.selectbox("Choose a dataset to analyze:", list(DATASETS.keys()))
        df = DATASETS[ds_key]()

        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        bool_cols = df.select_dtypes(include="bool").columns.tolist()

        # Overview
        st.markdown("### Step 1 — Dataset Overview")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Rows", f"{df.shape[0]:,}")
        c2.metric("Columns", df.shape[1])
        c3.metric("Numerical", len(num_cols))
        c4.metric("Categorical", len(cat_cols))
        c5.metric("Missing %", f"{df.isnull().mean().mean()*100:.1f}%")

        st.dataframe(df.describe(include="all"), use_container_width=True)

        # Distributions
        st.markdown("### Step 2 — Univariate Analysis")
        if num_cols:
            col_u = st.selectbox("Select numerical column:", num_cols, key="eda_uni")
            col_a, col_b = st.columns(2)
            with col_a:
                fig_hist = plot_distribution(df, col_u)
                st.plotly_chart(fig_hist, use_container_width=True)
            with col_b:
                fig_box = go.Figure(go.Box(y=df[col_u].dropna(), boxpoints="outliers",
                                           marker_color="#3b82f6", name=col_u))
                fig_box = fig_defaults(fig_box, f"{col_u} — Box Plot")
                st.plotly_chart(fig_box, use_container_width=True)

            # Stats
            s = df[col_u].describe()
            skew = df[col_u].skew()
            kurt = df[col_u].kurtosis()
            st.markdown(f"""
            | Stat | Value |
            |---|---|
            | Mean | {s['mean']:.4f} |
            | Median | {df[col_u].median():.4f} |
            | Std Dev | {s['std']:.4f} |
            | Skewness | {skew:.3f} {'(right-skewed ▶)' if skew > 0.5 else '(left-skewed ◀)' if skew < -0.5 else '(symmetric ↔)'} |
            | Kurtosis | {kurt:.3f} {'(heavy tails ↑)' if kurt > 1 else '(normal tails)'} |
            """)

        if cat_cols:
            col_c = st.selectbox("Select categorical column:", cat_cols, key="eda_cat")
            vc = df[col_c].value_counts().reset_index()
            vc.columns = [col_c, "count"]
            fig_bar = px.bar(vc, x=col_c, y="count", color=col_c,
                             color_discrete_sequence=px.colors.qualitative.Set2)
            fig_bar = fig_defaults(fig_bar, f"{col_c} — Value Counts")
            st.plotly_chart(fig_bar, use_container_width=True)

        # Bivariate
        st.markdown("### Step 3 — Bivariate Analysis")
        if len(num_cols) >= 2:
            c1, c2, c3 = st.columns(3)
            x_col = c1.selectbox("X axis:", num_cols, key="eda_bx")
            y_col = c2.selectbox("Y axis:", num_cols, index=min(1, len(num_cols)-1), key="eda_by")
            color_opt = c3.selectbox("Color by:", ["None"] + cat_cols, key="eda_bc")
            fig_sc = plot_scatter(df, x_col, y_col, None if color_opt == "None" else color_opt)
            st.plotly_chart(fig_sc, use_container_width=True)

            corr = df[x_col].corr(df[y_col])
            st.markdown(f"""
            <div class="box-info">
            Pearson correlation between <strong>{x_col}</strong> and <strong>{y_col}</strong>: <strong>{corr:.4f}</strong>
            {"— Strong positive relationship 📈" if corr > 0.7 else "— Strong negative relationship 📉" if corr < -0.7 else "— Moderate relationship" if abs(corr) > 0.4 else "— Weak/no linear relationship"}
            </div>
            """, unsafe_allow_html=True)

        # Correlation matrix
        st.markdown("### Step 4 — Correlation Matrix")
        if len(num_cols) >= 3:
            st.plotly_chart(plot_correlation_heatmap(df), use_container_width=True)

        # Target analysis
        st.markdown("### Step 5 — Target Variable Analysis")
        bool_and_cat = bool_cols + cat_cols
        if bool_and_cat and num_cols:
            target = st.selectbox("Select target/label column:", bool_and_cat, key="eda_tgt")
            feat = st.selectbox("Feature to compare:", num_cols, key="eda_feat")
            df_plot = df.copy()
            df_plot[target] = df_plot[target].astype(str)
            fig_tgt = plot_box(df_plot, target, feat)
            st.plotly_chart(fig_tgt, use_container_width=True)

    with tab3:
        st.markdown("### EDA Code — Full Template")
        nb_bytes = make_notebook("EDA — DS Workshop 2026", TOPIC_NOTEBOOKS.get("EDA", []))
        st.download_button("📓 Download as Jupyter Notebook", nb_bytes,
                           file_name="05_eda.ipynb", mime="application/json")
        st.markdown("---")
                st.code("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

df = pd.read_csv("data.csv")

# ── 1. Overview ────────────────────────────────────────────────────────
print(df.shape)
print(df.dtypes)
print(df.head())
df.info()
df.describe(include="all")

# ── 2. Missing values ──────────────────────────────────────────────────
missing = df.isnull().sum().sort_values(ascending=False)
missing_pct = (missing / len(df) * 100).round(2)
print(pd.concat([missing, missing_pct], axis=1, keys=["count", "pct"]))

# ── 3. Distributions ───────────────────────────────────────────────────
df.hist(figsize=(14, 10), bins=30, color="#3b82f6", edgecolor="white")
plt.tight_layout()
plt.show()

# Skewness
print(df.select_dtypes(include=np.number).skew().sort_values())

# ── 4. Categorical analysis ────────────────────────────────────────────
for col in df.select_dtypes(include="object").columns:
    print(f"\n{col}:")
    print(df[col].value_counts(normalize=True).head(10))

# ── 5. Correlation ─────────────────────────────────────────────────────
corr_matrix = df.select_dtypes(include=np.number).corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu", center=0)
plt.show()

# Target correlation
target_corr = corr_matrix["target"].sort_values(ascending=False)
print(target_corr)

# ── 6. Outlier detection ───────────────────────────────────────────────
df.boxplot(figsize=(14, 6))
plt.show()

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
outliers = ((df < Q1 - 1.5*IQR) | (df > Q3 + 1.5*IQR)).sum()
print("Outliers per column:", outliers)

# ── 7. Bivariate with target ───────────────────────────────────────────
# For classification: compare distributions by class
for col in df.select_dtypes(include=np.number).columns:
    df.boxplot(col, by="target")
    plt.show()

# ── 8. pandas-profiling (automated EDA report) ─────────────────────────
from ydata_profiling import ProfileReport
profile = ProfileReport(df, title="EDA Report", explorative=True)
profile.to_file("eda_report.html")
        """, language="python")

    with tab4:
        st.markdown("### Quiz — EDA")
        questions = [
            {"q": "A feature has skewness of 2.3. What does this indicate?",
             "options": ["Symmetric distribution", "Heavy left tail", "Heavy right tail", "Uniform distribution"],
             "answer": "Heavy right tail",
             "explanation": "Positive skewness = right-skewed = long right tail. Examples: salary, house prices, sensor failure times."},
            {"q": "Pearson correlation between two features is 0.02. What should you conclude?",
             "options": ["Strong positive relationship", "No relationship at all", "No LINEAR relationship (may have nonlinear)", "Negative relationship"],
             "answer": "No LINEAR relationship (may have nonlinear)",
             "explanation": "Pearson only detects linear relationships. Two variables can be strongly related (e.g., quadratic) yet show r≈0."},
            {"q": "During EDA you find feature A has correlation 0.98 with the target. What's the risk?",
             "options": ["Nothing, great feature!", "Data leakage — A may encode the target", "The model will overfit", "A should be log-transformed"],
             "answer": "Data leakage — A may encode the target",
             "explanation": "Near-perfect correlation often means the feature directly encodes the answer (leakage). Investigate before using."},
        ]
        for i, q in enumerate(questions):
            st.markdown(f"**Q{i+1}.** {q['q']}")
            choice = st.radio("", q["options"], key=f"quiz_05_q{i}", index=None, horizontal=True)
            if choice:
                if choice == q["answer"]:
                    st.markdown(f'<div class="box-success">✅ Correct! {q["explanation"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="box-warning">❌ Answer: **{q["answer"]}**. {q["explanation"]}</div>', unsafe_allow_html=True)
            st.markdown("")
        if "eda" not in st.session_state.completed_topics:
            if st.button("✅ Mark topic as complete"):
                st.session_state.completed_topics.append("eda")
                st.success("Topic marked complete!")
