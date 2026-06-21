import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from app.utils.datasets import DATASETS
from app.utils.notebook_export import make_notebook, TOPIC_NOTEBOOKS
from app.utils.visualizations import plot_distribution, fig_defaults
import plotly.graph_objects as go
import plotly.express as px

def render():
    st.markdown("""
    <div class="hero-banner" style="padding:1.5rem;">
      <h1 style="font-size:1.8rem;">⚙️ Data Processing</h1>
      <p>Transforming raw clean data into ML-ready features — encoding, scaling, pipelines</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📖 Explain", "🎨 Transform Lab", "💻 Code", "✅ Quiz"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### What is Data Processing?")
            st.markdown("""
Data processing converts **clean raw data** into a **model-ready format**.

Even perfectly clean data can't go straight into ML algorithms:
- Text labels must become numbers
- Different-scale features confuse distance-based models
- Date columns need to be decomposed
- Skewed distributions need transformation
            """)
            st.code("""
PROCESSING PIPELINE
───────────────────────────────────────────────────
Raw Clean Data
     │
     ├── Encode categoricals      ("Steel" → 0, "Al" → 1)
     ├── Scale numericals         (0-1 range or z-score)
     ├── Decompose datetimes      (hour, weekday, month)
     ├── Transform distributions  (log, sqrt, Box-Cox)
     ├── Create interaction terms (A × B features)
     └── Build feature matrix X   → Ready for ML ✓
            """, language="text")

        with col2:
            st.markdown("### Encoding Strategies")
            enc = {
                "Method": ["Label Encoding", "One-Hot Encoding", "Target Encoding", "Ordinal Encoding", "Binary Encoding"],
                "When to use": ["Tree models (RF, XGB)", "Linear models, few categories", "High-cardinality + target", "Known order exists", "High cardinality"],
                "Example": ["Steel→0, Al→1", "[1,0], [0,1]", "Mean target per cat", "Low→0, Med→1, High→2", "3→011"],
            }
            st.dataframe(pd.DataFrame(enc), use_container_width=True, hide_index=True)

            st.markdown("### Scaling Strategies")
            sc = {
                "Method": ["StandardScaler", "MinMaxScaler", "RobustScaler", "Log Transform"],
                "Formula": ["(x-μ)/σ", "(x-min)/(max-min)", "(x-Q2)/(Q3-Q1)", "log(x+1)"],
                "Use When": ["Normal dist, no outliers", "Bounded range needed", "Outliers present", "Right-skewed data"],
            }
            st.dataframe(pd.DataFrame(sc), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("""
        <div class="box-info">
        <strong>🧠 AI Perspective:</strong> Neural networks are extremely sensitive to scale.
        Unscaled inputs → vanishing or exploding gradients → training failure.
        Always fit scalers on TRAINING data only — apply to test data.
        Fitting on the full dataset causes <strong>data leakage</strong>.
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("### Interactive Processing Lab")
        ds_key = st.selectbox("Choose dataset:", list(DATASETS.keys()))
        df = DATASETS[ds_key]().copy()

        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include="object").columns.tolist()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Scale Numerical Features")
            scale_cols = st.multiselect("Select columns to scale:", num_cols, default=num_cols[:2] if len(num_cols)>=2 else num_cols)
            scale_method = st.radio("Scaling method:", ["StandardScaler", "MinMaxScaler", "RobustScaler", "Log(x+1)"])

        with col2:
            st.markdown("#### Encode Categorical Features")
            encode_cols = st.multiselect("Select columns to encode:", cat_cols, default=cat_cols[:1] if cat_cols else [])
            encode_method = st.radio("Encoding method:", ["One-Hot Encoding", "Label Encoding", "Ordinal Encoding"])

        if st.button("⚙️ Apply Transformations", type="primary"):
            df_proc = df.copy()
            from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder

            if scale_cols:
                if scale_method == "StandardScaler":
                    scaler = StandardScaler()
                    df_proc[scale_cols] = scaler.fit_transform(df_proc[scale_cols].fillna(df_proc[scale_cols].median()))
                elif scale_method == "MinMaxScaler":
                    scaler = MinMaxScaler()
                    df_proc[scale_cols] = scaler.fit_transform(df_proc[scale_cols].fillna(df_proc[scale_cols].median()))
                elif scale_method == "RobustScaler":
                    scaler = RobustScaler()
                    df_proc[scale_cols] = scaler.fit_transform(df_proc[scale_cols].fillna(df_proc[scale_cols].median()))
                elif scale_method == "Log(x+1)":
                    for col in scale_cols:
                        df_proc[col] = np.log1p(df_proc[col].clip(lower=0))

            if encode_cols:
                if encode_method == "One-Hot Encoding":
                    df_proc = pd.get_dummies(df_proc, columns=encode_cols, drop_first=False)
                elif encode_method == "Label Encoding":
                    le = LabelEncoder()
                    for col in encode_cols:
                        df_proc[col] = le.fit_transform(df_proc[col].astype(str))
                elif encode_method == "Ordinal Encoding":
                    for col in encode_cols:
                        cats = sorted(df_proc[col].dropna().unique())
                        mapping = {cat: i for i, cat in enumerate(cats)}
                        df_proc[col] = df_proc[col].map(mapping)

            st.markdown("### Processed Data")
            col1, col2, col3 = st.columns(3)
            col1.metric("Original columns", df.shape[1])
            col2.metric("New columns", df_proc.shape[1])
            col3.metric("New features added", df_proc.shape[1] - df.shape[1])

            st.dataframe(df_proc.head(15), use_container_width=True)

            if scale_cols and len(scale_cols) >= 1:
                col_plot = scale_cols[0]
                if col_plot in df.columns and col_plot in df_proc.columns:
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(x=df[col_plot].dropna(), name="Before", opacity=0.7, marker_color="#ef4444"))
                    fig.add_trace(go.Histogram(x=df_proc[col_plot].dropna(), name="After", opacity=0.7, marker_color="#10b981"))
                    fig.update_layout(barmode="overlay")
                    fig = fig_defaults(fig, f"{col_plot}: Before vs After {scale_method}")
                    st.plotly_chart(fig, use_container_width=True)

            st.download_button("⬇ Download Processed CSV", df_proc.to_csv(index=False),
                               file_name="processed_data.csv", mime="text/csv")

    with tab3:
        st.markdown("### Processing Code Reference")
        nb_bytes = make_notebook("Data Processing — DS Workshop 2026", TOPIC_NOTEBOOKS.get("Data Processing", []))
        st.download_button("📓 Download as Jupyter Notebook", nb_bytes,
                           file_name="04_data_processing.ipynb", mime="application/json")
        st.markdown("---")
                st.code("""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

df = pd.read_parquet("clean_data.parquet")

# ── 1. Date decomposition ─────────────────────────────────────────────
df["hour"]      = df["timestamp"].dt.hour
df["dayofweek"] = df["timestamp"].dt.dayofweek   # Mon=0, Sun=6
df["month"]     = df["timestamp"].dt.month
df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

# ── 2. Encoding ───────────────────────────────────────────────────────
# One-hot (creates new columns)
df = pd.get_dummies(df, columns=["material", "shift"], drop_first=True)

# Label encode (in-place)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["severity_code"] = le.fit_transform(df["severity"])

# ── 3. Scaling ─────────────────────────────────────────────────────────
numeric_features = ["temperature", "pressure", "vibration"]
scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# ── 4. Sklearn Pipeline (best practice) ──────────────────────────────
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

numeric_features = ["temperature", "pressure", "vibration"]
categorical_features = ["material", "shift", "line_id"]

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

# Use inside a full ML pipeline
from sklearn.ensemble import RandomForestClassifier
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier",   RandomForestClassifier(n_estimators=100)),
])

pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_test)   # preprocessing happens automatically!
        """, language="python")

    with tab4:
        st.markdown("### Quiz — Data Processing")
        questions = [
            {"q": "You apply StandardScaler to the full dataset before train/test split. What's the problem?",
             "options": ["Nothing, it's fine", "Data leakage — test stats contaminate the scaler", "StandardScaler only works on training data", "It makes the data non-normal"],
             "answer": "Data leakage — test stats contaminate the scaler",
             "explanation": "Fitting the scaler on all data means the model has 'seen' test statistics during training. Fit on train, transform on test."},
            {"q": "Which encoding is best for 'City' with 500 unique values in a tree model?",
             "options": ["One-hot encoding", "Label encoding", "Target encoding", "Drop the column"],
             "answer": "Target encoding",
             "explanation": "One-hot → 500 sparse columns. Label encoding → arbitrary order. Target encoding compresses to 1 meaningful numeric column."},
            {"q": "Your feature 'salary' is heavily right-skewed. Which transformation helps?",
             "options": ["StandardScaler", "One-hot encoding", "Log transform", "Binary encoding"],
             "answer": "Log transform",
             "explanation": "Log(x+1) compresses right tails and makes skewed distributions more normal — great for salary, revenue, count features."},
        ]
        for i, q in enumerate(questions):
            st.markdown(f"**Q{i+1}.** {q['q']}")
            choice = st.radio("", q["options"], key=f"quiz_04_q{i}", index=None, horizontal=True)
            if choice:
                if choice == q["answer"]:
                    st.markdown(f'<div class="box-success">✅ Correct! {q["explanation"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="box-warning">❌ Answer: **{q["answer"]}**. {q["explanation"]}</div>', unsafe_allow_html=True)
            st.markdown("")
        if "data_processing" not in st.session_state.completed_topics:
            if st.button("✅ Mark topic as complete"):
                st.session_state.completed_topics.append("data_processing")
                st.success("Topic marked complete!")
