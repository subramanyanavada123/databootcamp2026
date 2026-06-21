import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from app.utils.datasets import DATASETS
from app.utils.notebook_export import make_notebook, TOPIC_NOTEBOOKS
from app.utils.visualizations import (
    plot_distribution, plot_correlation_heatmap, plot_missing_heatmap,
    plot_box, fig_defaults
)
import plotly.express as px

def render():
    st.markdown("""
    <div class="hero-banner" style="padding:1.5rem;">
      <h1 style="font-size:1.8rem;">📊 Understanding Data</h1>
      <p>What data IS, how it's structured, and why quality matters</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📖 Explain", "🎨 Visualize", "💻 Code", "🧪 Explore", "✅ Quiz"])

    # ── TAB 1: EXPLAIN ────────────────────────────────────────────────────────
    with tab1:
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown("### What is Data?")
            st.markdown("""
**Simple:** Data is a **recorded observation** of the real world.

**Engineering:** Data is a discrete or continuous signal sampled from a system,
encoded into a storable, transmittable format with a defined schema,
resolution, and sampling rate.

> Every time something happens — a temperature changes, a user clicks,
> an engine vibrates — if you **capture that moment**, you have data.
            """)

            st.markdown("### The Data Taxonomy")
            st.code("""
DATA
 ├── STRUCTURED
 │    ├── Tabular (rows × columns)  ← Excel, SQL, CSV
 │    ├── Time Series               ← Sensor logs, stock prices
 │    └── Relational                ← Databases, joined tables
 │
 ├── SEMI-STRUCTURED
 │    ├── JSON / XML                ← APIs, config files
 │    └── Log files                 ← Server logs, event streams
 │
 └── UNSTRUCTURED
      ├── Text                      ← Documents, chat, reports
      ├── Images                    ← Satellite, medical scans
      ├── Audio                     ← Voice, vibration signals
      └── Video                     ← Surveillance, dashcams
            """, language="text")

        with col2:
            st.markdown("### Data Types")
            type_data = {
                "Type": ["Continuous", "Discrete", "Nominal", "Ordinal", "Boolean", "Datetime", "Text"],
                "Description": ["Any range value", "Countable integers", "Labels, no order", "Labels + order", "True/False", "Timestamp", "Free-form chars"],
                "Example": ["73.4°C", "3 defects", "Steel/Al/Cu", "Low/Med/High", "Valve open", "2026-06-21", "Bearing noise"],
            }
            st.dataframe(pd.DataFrame(type_data), use_container_width=True, hide_index=True)

            st.markdown("### Data Quality Dimensions")
            quality = {
                "Dimension": ["Completeness", "Accuracy", "Consistency", "Timeliness", "Validity", "Uniqueness"],
                "Meaning": ["No missing readings", "Calibrated sensors", "Same units everywhere", "Real-time vs batch", "No -999°C values", "No duplicate logs"],
            }
            st.dataframe(pd.DataFrame(quality), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### Industry Example — Aircraft Engine Data Stream")
        st.markdown("""
        Rolls-Royce Trent engines transmit **~500 parameters × 4 engines** every second during flight.
        That's **57.6 million data points per 8-hour flight**, processed in real-time via satellite for predictive maintenance.
        """)

        engine_example = pd.DataFrame({
            "Parameter":    ["EGT (Exhaust Temp)", "N1 Fan Speed", "Fuel Flow", "Vibration Level", "Engine ID", "Alert Triggered", "Severity", "Timestamp"],
            "Type":         ["Continuous", "Continuous", "Continuous", "Continuous", "Nominal", "Boolean", "Ordinal", "Datetime"],
            "Sample Value": ["642.7 °C", "94.3 %", "2847 kg/hr", "0.42 in/sec", "ENG-L-001", "False", "Normal", "2026-06-21 09:14"],
        })
        st.dataframe(engine_example, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("""
        <div class="box-info">
        <strong>🧠 AI Perspective:</strong> ML models ONLY understand numbers.
        Every string, category, and boolean must be converted to a numeric vector before training.
        This translation is called <strong>Feature Engineering</strong> (Stage 6).
        <br/><br/>
        <code>[642.7, 94.3, 0.42, 2847, 0, 1, 0]</code>
        ← what an AI "sees" instead of the table above.
        </div>
        """, unsafe_allow_html=True)

    # ── TAB 2: VISUALIZE ─────────────────────────────────────────────────────
    with tab2:
        st.markdown("### Explore a Real Industry Dataset")

        ds_choice = st.selectbox("Choose a dataset:", list(DATASETS.keys()))
        df = DATASETS[ds_choice]()

        col1, col2, col3, col4 = st.columns(4)
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        with col1:
            st.metric("Rows", f"{df.shape[0]:,}")
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Numeric cols", len(num_cols))
        with col4:
            miss = df.isnull().sum().sum()
            st.metric("Missing values", miss)

        st.markdown("#### Raw Data Preview")
        st.dataframe(df.head(20), use_container_width=True)

        st.markdown("#### Data Types")
        dtype_df = pd.DataFrame({"Column": df.columns, "Type": df.dtypes.astype(str).values,
                                  "Non-null": df.count().values, "Missing": df.isnull().sum().values})
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)

        if num_cols:
            st.markdown("#### Distribution Explorer")
            col_choice = st.selectbox("Select column:", num_cols)
            cat_cols = df.select_dtypes(include="object").columns.tolist()
            color_by = st.selectbox("Color by (optional):", ["None"] + cat_cols)
            color_col = None if color_by == "None" else color_by
            st.plotly_chart(plot_distribution(df, col_choice, color_col), use_container_width=True)

        st.markdown("#### Missing Values Heatmap")
        fig_miss = plot_missing_heatmap(df)
        if fig_miss:
            st.plotly_chart(fig_miss, use_container_width=True)
        else:
            st.markdown('<div class="box-success">✅ No missing values in this dataset!</div>', unsafe_allow_html=True)

        if len(num_cols) >= 3:
            st.markdown("#### Correlation Matrix")
            st.plotly_chart(plot_correlation_heatmap(df), use_container_width=True)

    # ── TAB 3: CODE ───────────────────────────────────────────────────────────
    with tab3:
        st.markdown("### Python Code Examples")
        nb_bytes = make_notebook("Understanding Data — DS Workshop 2026", TOPIC_NOTEBOOKS["Understanding Data"])
        st.download_button("📓 Download as Jupyter Notebook", nb_bytes,
                           file_name="01_understanding_data.ipynb", mime="application/json")
        st.markdown("---")

        st.markdown("#### 1. Creating and inspecting a DataFrame")
        st.code("""
import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000

engine_data = pd.DataFrame({
    'timestamp':       pd.date_range('2026-06-21', periods=n, freq='1s'),
    'engine_id':       np.random.choice(['ENG-L-001', 'ENG-R-001'], n),
    'egt_celsius':     np.random.normal(640, 15, n),        # Continuous
    'n1_fan_speed':    np.random.normal(94, 2, n),          # Continuous
    'vibration':       np.random.exponential(0.4, n),       # Continuous
    'alert_active':    np.random.choice([True, False],      # Boolean
                           n, p=[0.05, 0.95]),
    'severity':        np.random.choice(                    # Ordinal
                           ['Normal','Caution','Warning'],
                           n, p=[0.90, 0.08, 0.02])
})

print(engine_data.shape)          # (1000, 7)
print(engine_data.dtypes)         # column types
print(engine_data.describe())     # statistics
print(engine_data.head())         # first 5 rows
        """, language="python")

        st.markdown("#### 2. Checking Data Quality")
        st.code("""
# Missing values
print(df.isnull().sum())
print(f"Total missing: {df.isnull().sum().sum()}")
print(f"Missing %: {df.isnull().mean() * 100}")

# Duplicates
print(f"Duplicate rows: {df.duplicated().sum()}")

# Value ranges (validity check)
print(df.describe())

# Check categorical values
print(df['severity'].value_counts())
print(df['engine_id'].nunique(), "unique engines")
        """, language="python")

        st.markdown("#### 3. Data Type Conversion")
        st.code("""
# Convert string to categorical (memory efficient)
df['severity'] = pd.Categorical(df['severity'],
    categories=['Normal', 'Caution', 'Warning'],
    ordered=True)

# Convert string to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Encode boolean as int for ML
df['alert_int'] = df['alert_active'].astype(int)

# Memory usage before/after
print(df.memory_usage(deep=True))
        """, language="python")

        st.markdown("#### 4. Run it live — Manufacturing QC Dataset")
        if st.button("▶ Run Analysis"):
            from app.utils.datasets import get_manufacturing_dataset
            df_mfg = get_manufacturing_dataset()
            st.write("**Shape:**", df_mfg.shape)
            st.write("**Data Types:**")
            st.dataframe(df_mfg.dtypes.reset_index().rename(columns={"index": "Column", 0: "Type"}), hide_index=True)
            st.write("**Statistics:**")
            st.dataframe(df_mfg.describe(), use_container_width=True)
            st.write("**QC Pass Rate:**", f"{df_mfg['passed_qc'].mean()*100:.1f}%")
            st.write("**Failure rate by material:**")
            st.dataframe(
                df_mfg.groupby("material")["passed_qc"]
                .agg(lambda x: f"{(~x).mean()*100:.1f}%")
                .reset_index().rename(columns={"passed_qc": "Failure Rate"}),
                hide_index=True
            )

    # ── TAB 4: EXPLORE ────────────────────────────────────────────────────────
    with tab4:
        st.markdown("### Interactive Data Explorer")
        st.markdown("Load any dataset and explore it freely:")

        ds_choice2 = st.selectbox("Dataset:", list(DATASETS.keys()), key="explore_ds")
        df2 = DATASETS[ds_choice2]()

        exp1 = st.expander("📋 Full Dataset", expanded=False)
        with exp1:
            st.dataframe(df2, use_container_width=True)
            st.download_button("⬇ Download CSV", df2.to_csv(index=False),
                               file_name="dataset.csv", mime="text/csv")

        exp2 = st.expander("📊 Statistical Summary", expanded=True)
        with exp2:
            st.dataframe(df2.describe(include="all"), use_container_width=True)

        num_cols2 = df2.select_dtypes(include=np.number).columns.tolist()
        cat_cols2 = df2.select_dtypes(include="object").columns.tolist()

        exp3 = st.expander("🔭 Scatter Plot — Find Relationships")
        with exp3:
            if len(num_cols2) >= 2:
                c1, c2, c3 = st.columns(3)
                x_col = c1.selectbox("X axis:", num_cols2, key="sc_x")
                y_col = c2.selectbox("Y axis:", num_cols2, index=min(1, len(num_cols2)-1), key="sc_y")
                color_opt = c3.selectbox("Color by:", ["None"] + cat_cols2, key="sc_c")
                from app.utils.visualizations import plot_scatter
                st.plotly_chart(
                    plot_scatter(df2, x_col, y_col, None if color_opt=="None" else color_opt),
                    use_container_width=True
                )

        exp4 = st.expander("📦 Box Plot — Distributions by Category")
        with exp4:
            if num_cols2 and cat_cols2:
                c1, c2 = st.columns(2)
                y_b = c1.selectbox("Numeric:", num_cols2, key="bx_y")
                x_b = c2.selectbox("Category:", cat_cols2, key="bx_x")
                st.plotly_chart(plot_box(df2, x_b, y_b), use_container_width=True)

    # ── TAB 5: QUIZ ───────────────────────────────────────────────────────────
    with tab5:
        st.markdown("### Knowledge Check — Understanding Data")

        questions = [
            {
                "q": "A telecom company stores call duration in seconds. What data type is this?",
                "options": ["Nominal", "Ordinal", "Continuous", "Boolean"],
                "answer": "Continuous",
                "explanation": "Call duration can be any non-negative real number — it's continuous numerical data."
            },
            {
                "q": "'Signal strength: Excellent / Good / Poor / None' — what type of categorical variable?",
                "options": ["Nominal", "Ordinal", "Discrete", "Continuous"],
                "answer": "Ordinal",
                "explanation": "There is a meaningful ORDER: Excellent > Good > Poor > None. That makes it ordinal."
            },
            {
                "q": "You receive temperature sensor data with 15% missing values. Which quality dimension is violated?",
                "options": ["Accuracy", "Validity", "Completeness", "Uniqueness"],
                "answer": "Completeness",
                "explanation": "Completeness means all expected data is present. 15% missing is a completeness issue."
            },
            {
                "q": "An IoT device sends the same heartbeat packet twice. Which dimension does this violate?",
                "options": ["Timeliness", "Uniqueness", "Consistency", "Validity"],
                "answer": "Uniqueness",
                "explanation": "Duplicate records violate the Uniqueness dimension — each event should appear exactly once."
            },
        ]

        score = 0
        answered = 0
        for i, q in enumerate(questions):
            st.markdown(f"**Q{i+1}.** {q['q']}")
            key = f"quiz_01_q{i}"
            choice = st.radio("", q["options"], key=key, index=None, horizontal=True)
            if choice:
                answered += 1
                if choice == q["answer"]:
                    st.markdown(f'<div class="box-success">✅ Correct! {q["explanation"]}</div>', unsafe_allow_html=True)
                    score += 1
                else:
                    st.markdown(f'<div class="box-warning">❌ Incorrect. Correct answer: **{q["answer"]}**. {q["explanation"]}</div>', unsafe_allow_html=True)
            st.markdown("")

        if answered == len(questions):
            st.markdown("---")
            pct = int(score / len(questions) * 100)
            if pct == 100:
                st.balloons()
                st.markdown(f'<div class="box-success">🏆 Perfect score! {score}/{len(questions)} ({pct}%) — You\'re ready for Stage 2!</div>', unsafe_allow_html=True)
            elif pct >= 75:
                st.markdown(f'<div class="box-success">✅ Great! {score}/{len(questions)} ({pct}%) — Solid understanding!</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="box-warning">📚 {score}/{len(questions)} ({pct}%) — Review the Explain tab and try again!</div>', unsafe_allow_html=True)

            st.session_state.quiz_score += score
            st.session_state.quiz_total += len(questions)
            if "understanding_data" not in st.session_state.completed_topics:
                st.session_state.completed_topics.append("understanding_data")
