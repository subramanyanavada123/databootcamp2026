import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from app.utils.datasets import DATASETS
from app.utils.visualizations import fig_defaults
from app.utils.notebook_export import make_notebook, TOPIC_NOTEBOOKS

# ── Requirements per chart type ────────────────────────────────────────────────
CHART_REQUIREMENTS = {
    "📊 Histogram": {
        "needs": ["numeric"],
        "why_not": {
            "numeric": "Histograms show the **distribution of a numerical column** — they need at least one numeric column to bin into bars. This dataset has no numeric columns.",
        },
        "tip": "Try the **🔩 Manufacturing QC** or **✈️ Aircraft Engine** datasets which have continuous measurements.",
    },
    "📦 Box Plot": {
        "needs": ["numeric"],
        "why_not": {
            "numeric": "Box plots summarise a **numerical distribution** (median, quartiles, outliers). They require at least one numeric column.",
        },
        "tip": "Any dataset with sensor readings or measurements will work here.",
    },
    "🔵 Scatter Plot": {
        "needs": ["numeric_2"],
        "why_not": {
            "numeric_2": "Scatter plots map **two numerical variables** onto X and Y axes to reveal correlations. This dataset needs at least 2 numeric columns.",
        },
        "tip": "Datasets with multiple sensor readings (temperature, pressure, vibration) are ideal.",
    },
    "📈 Line Chart": {
        "needs": ["datetime", "numeric"],
        "why_not": {
            "datetime": "Line charts show **trends over time** — they need a datetime column for the X axis. This dataset has no timestamp or date column.",
            "numeric": "Line charts also need a numeric Y axis.",
        },
        "tip": "Use the **🌡️ IoT Sensors** or **✈️ Aircraft Engine** datasets which have timestamp columns.",
    },
    "📊 Bar Chart": {
        "needs": ["categorical", "numeric"],
        "why_not": {
            "categorical": "Bar charts compare **values across categories** — they need a categorical (text) column to group by.",
            "numeric": "Bar charts also need a numeric column to aggregate (sum, mean, etc.).",
        },
        "tip": "The **🔩 Manufacturing QC** dataset has material, surface_finish columns perfect for grouping.",
    },
    "🌡️ Heatmap": {
        "needs": ["numeric_3"],
        "why_not": {
            "numeric_3": "Correlation heatmaps need **at least 3 numeric columns** to compute a meaningful correlation matrix.",
        },
        "tip": "Datasets with multiple sensor readings work best.",
    },
    "🫧 Bubble Chart": {
        "needs": ["numeric_3"],
        "why_not": {
            "numeric_3": "Bubble charts need **3 numeric columns**: X axis, Y axis, and bubble size.",
        },
        "tip": "Use datasets with ≥3 measurement columns.",
    },
    "🎻 Violin Plot": {
        "needs": ["numeric"],
        "why_not": {
            "numeric": "Violin plots show a **distribution shape** — they need at least one numeric column.",
        },
        "tip": "Works well with any measurement dataset.",
    },
    "🥧 Sunburst": {
        "needs": ["categorical_2"],
        "why_not": {
            "categorical_2": "Sunburst charts show **hierarchical categorical data** — they need at least 2 different categorical (text) columns to create parent→child segments.",
        },
        "tip": "The **🔩 Manufacturing QC** dataset has material + surface_finish which form a natural hierarchy.",
    },
    "📉 Area Chart": {
        "needs": ["datetime", "numeric"],
        "why_not": {
            "datetime": "Area charts show **cumulative trends over time** — they need a datetime column. This dataset has no date/time column.",
            "numeric": "Area charts also need a numeric Y axis.",
        },
        "tip": "Use the **🌡️ IoT Sensors** dataset which has 5-minute interval timestamps.",
    },
}


def _check_requirements(chart_type: str, num_cols: list, cat_cols: list, date_cols: list):
    """Returns (ok: bool, reason: str, tip: str)"""
    req = CHART_REQUIREMENTS.get(chart_type, {})
    needs = req.get("needs", [])
    why_not = req.get("why_not", {})
    tip = req.get("tip", "")

    if "numeric" in needs and len(num_cols) == 0:
        return False, why_not.get("numeric", ""), tip
    if "numeric_2" in needs and len(num_cols) < 2:
        return False, f"This chart needs **at least 2 numeric columns** — this dataset only has {len(num_cols)}. Scatter plots map two measurements against each other to reveal correlation.", tip
    if "numeric_3" in needs and len(num_cols) < 3:
        return False, f"This chart needs **at least 3 numeric columns** — this dataset only has {len(num_cols)}.", tip
    if "categorical" in needs and len(cat_cols) == 0:
        return False, why_not.get("categorical", ""), tip
    if "categorical_2" in needs and len(cat_cols) < 2:
        return False, f"Sunburst charts need **at least 2 categorical columns** — this dataset only has {len(cat_cols)}.", tip
    if "datetime" in needs and len(date_cols) == 0:
        return False, why_not.get("datetime", ""), tip

    return True, "", ""


def _not_possible_card(reason: str, tip: str, chart_type: str, num_cols, cat_cols, date_cols):
    st.markdown(f"""
    <div style="background:#1c1917; border:1px solid #78350f; border-radius:12px; padding:1.5rem; margin-top:1rem;">
      <div style="font-size:1.5rem; margin-bottom:0.5rem;">🚫</div>
      <div style="font-weight:700; color:#fde68a; font-size:1rem; margin-bottom:0.75rem;">
        Cannot render {chart_type} with this dataset
      </div>
      <div style="color:#d4a574; line-height:1.6; margin-bottom:1rem;">{reason}</div>
      <div style="border-top:1px solid #78350f; padding-top:0.75rem; font-size:0.85rem; color:#92400e;">
        <strong style="color:#f59e0b;">💡 Tip:</strong> {tip}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Show what this dataset CAN do
    possible = []
    for ct, req in CHART_REQUIREMENTS.items():
        ok, _, _ = _check_requirements(ct, num_cols, cat_cols, date_cols)
        if ok:
            possible.append(ct)
    if possible:
        st.markdown("**This dataset works with:**")
        cols = st.columns(min(len(possible), 4))
        for i, ct in enumerate(possible):
            cols[i % 4].markdown(f'<span class="badge badge-blue">{ct}</span>', unsafe_allow_html=True)

    # Educational context
    with st.expander("📚 Why does this matter? (Data Science Insight)"):
        st.markdown(f"""
**Chart–data compatibility is a core data literacy skill.**

Different charts encode different *types of information*:

| Chart | Encodes | Requires |
|---|---|---|
| Histogram | Distribution shape | 1 numeric |
| Box Plot | Summary statistics | 1 numeric |
| Scatter | Relationship (correlation) | 2 numerics |
| Line/Area | Trend over time | 1 datetime + 1 numeric |
| Bar | Category comparison | 1 categorical + 1 numeric |
| Heatmap | Pairwise correlation | ≥3 numerics |
| Sunburst | Hierarchy / part-of-whole | ≥2 categoricals |
| Bubble | 3-way relationship | 3 numerics |

**This dataset has:** {len(num_cols)} numeric, {len(cat_cols)} categorical, {len(date_cols)} datetime columns.

Choosing the wrong chart doesn't just look bad — it can **actively mislead** your audience.
A line chart on non-time data implies a trend that doesn't exist.
A pie chart with 12 slices is unreadable.
""")


def render():
    st.markdown("""
    <div class="hero-banner" style="padding:1.5rem;">
      <h1 style="font-size:1.8rem;">📈 Data Visualization</h1>
      <p>Making data speak — chart selection, storytelling, interactive dashboards</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📖 Explain", "🎨 Chart Studio", "💻 Code + 📓 Notebook", "✅ Quiz"])

    # ── TAB 1: EXPLAIN ────────────────────────────────────────────────────────
    with tab1:
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown("### Why Visualization?")
            st.markdown("""
Visualization is the bridge between numbers and human understanding.
The best model in the world is worthless if no one believes its insights.

> "A picture is worth a thousand rows of data."

Good visualization:
- Reveals patterns invisible in tables
- Communicates uncertainty honestly
- Guides stakeholders to decisions
- Validates your analysis
            """)
            st.code("""
CHART SELECTION GUIDE
────────────────────────────────────────────────────────────────
 QUESTION                        CHART TYPE
 ─────────────────────────────────────────────────────────────
 Distribution of one variable    → Histogram / KDE / Box
 Compare distributions (groups)  → Grouped Box / Violin
 Relationship between 2 nums     → Scatter / Bubble
 Trend over time                 → Line / Area
 Category comparison             → Bar / Lollipop
 Part-of-whole                   → Pie / Treemap (use sparingly)
 Correlation matrix              → Heatmap
 Hierarchical categories         → Sunburst / Treemap
 Multivariate relationships      → Pair plot / Parallel coords
 Model performance               → ROC curve / Confusion matrix
            """, language="text")

        with col2:
            st.markdown("### Visualization Principles")
            st.markdown("""
**Do:**
- Title tells the insight, not just what's plotted
- Label axes with units
- Use color purposefully (not decoratively)
- Show uncertainty (error bars, confidence intervals)
- Keep it simple — one chart, one message

**Don't:**
- 3D pie charts (ever)
- Dual y-axes without care
- Truncated y-axes to exaggerate changes
- Rainbow color scales (use diverging or sequential)
- Overload with data points
            """)

            st.markdown("### Chart × Data Compatibility")
            compat = {
                "Chart": ["Histogram", "Box", "Scatter", "Line/Area", "Bar", "Heatmap", "Sunburst", "Bubble"],
                "Needs": ["1 numeric", "1 numeric", "2 numeric", "datetime + numeric", "cat + numeric", "3+ numeric", "2+ categorical", "3 numeric"],
            }
            st.dataframe(pd.DataFrame(compat), use_container_width=True, hide_index=True)

    # ── TAB 2: CHART STUDIO ───────────────────────────────────────────────────
    with tab2:
        st.markdown("### Interactive Chart Studio")

        col_ds, col_ct = st.columns(2)
        ds_key = col_ds.selectbox("Dataset:", list(DATASETS.keys()))
        chart_type = col_ct.selectbox("Chart type:", list(CHART_REQUIREMENTS.keys()))

        df = DATASETS[ds_key]()
        num_cols  = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols  = df.select_dtypes(include="object").columns.tolist()
        date_cols = df.select_dtypes(include="datetime").columns.tolist()

        # Show dataset summary chips
        st.markdown(
            f'<span class="badge badge-blue">{len(num_cols)} numeric</span> '
            f'<span class="badge badge-green">{len(cat_cols)} categorical</span> '
            f'<span class="badge badge-orange">{len(date_cols)} datetime</span> '
            f'<span class="badge badge-purple">{len(df):,} rows</span>',
            unsafe_allow_html=True
        )
        st.markdown("")

        # Check compatibility first
        ok, reason, tip = _check_requirements(chart_type, num_cols, cat_cols, date_cols)

        if not ok:
            _not_possible_card(reason, tip, chart_type, num_cols, cat_cols, date_cols)
            return

        # ── Render the chart ──────────────────────────────────────────────────
        fig = None

        if "Histogram" in chart_type:
            c1, c2, c3 = st.columns(3)
            col = c1.selectbox("Column:", num_cols)
            bins = c2.slider("Bins:", 10, 100, 30)
            color = c3.selectbox("Color by:", ["None"] + cat_cols)
            fig = px.histogram(df, x=col, nbins=bins,
                               color=None if color == "None" else color,
                               color_discrete_sequence=px.colors.qualitative.Set2,
                               barmode="overlay", opacity=0.8)
            fig = fig_defaults(fig, f"Distribution of {col}")

        elif "Box" in chart_type:
            c1, c2 = st.columns(2)
            y_col = c1.selectbox("Numeric:", num_cols)
            x_col = c2.selectbox("Group by:", ["None"] + cat_cols)
            if x_col == "None":
                fig = px.box(df, y=y_col, color_discrete_sequence=["#3b82f6"])
            else:
                fig = px.box(df, x=x_col, y=y_col, color=x_col,
                             color_discrete_sequence=px.colors.qualitative.Set2)
            fig = fig_defaults(fig, f"{y_col} Distribution")

        elif "Scatter" in chart_type:
            c1, c2, c3, c4 = st.columns(4)
            x = c1.selectbox("X:", num_cols)
            y = c2.selectbox("Y:", [c for c in num_cols if c != x] or num_cols,
                             index=0)
            color = c3.selectbox("Color:", ["None"] + cat_cols)
            size  = c4.selectbox("Size:", ["None"] + num_cols)
            # size column can't have negatives or NaN
            size_col = None
            if size != "None":
                if df[size].min() >= 0 and df[size].isnull().sum() == 0:
                    size_col = size
                else:
                    st.caption(f"⚠️ Size column `{size}` has negative/missing values — size disabled.")
            fig = px.scatter(df, x=x, y=y,
                             color=None if color == "None" else color,
                             size=size_col,
                             color_discrete_sequence=px.colors.qualitative.Set2,
                             opacity=0.7, hover_data=df.columns.tolist()[:5])
            fig = fig_defaults(fig, f"{y} vs {x}", height=500)

        elif "Line" in chart_type:
            c1, c2, c3 = st.columns(3)
            x = c1.selectbox("Time column:", date_cols)
            y = c2.selectbox("Value:", num_cols)
            color = c3.selectbox("Split by:", ["None"] + cat_cols)
            fig = px.line(df.sort_values(x), x=x, y=y,
                          color=None if color == "None" else color,
                          color_discrete_sequence=px.colors.qualitative.Set2)
            fig = fig_defaults(fig, f"{y} over Time", height=400)

        elif "Bar" in chart_type:
            c1, c2, c3 = st.columns(3)
            x   = c1.selectbox("Category:", cat_cols)
            y   = c2.selectbox("Value:", num_cols)
            agg = c3.selectbox("Aggregate:", ["mean", "sum", "count", "median"])
            df_agg = df.groupby(x)[y].agg(agg).reset_index()
            fig = px.bar(df_agg, x=x, y=y, color=x,
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig = fig_defaults(fig, f"{agg.title()} of {y} by {x}")

        elif "Heatmap" in chart_type:
            selected = st.multiselect("Columns for correlation:", num_cols, default=num_cols[:6])
            if len(selected) < 2:
                st.warning("Select at least 2 columns.")
            else:
                corr = df[selected].corr()
                fig = go.Figure(go.Heatmap(
                    z=corr.values, x=corr.columns, y=corr.index,
                    colorscale="RdBu", zmid=0,
                    text=corr.round(2).values, texttemplate="%{text}",
                ))
                fig = fig_defaults(fig, "Correlation Heatmap", height=500)

        elif "Bubble" in chart_type:
            c1, c2, c3, c4 = st.columns(4)
            x     = c1.selectbox("X:", num_cols)
            y     = c2.selectbox("Y:", [c for c in num_cols if c != x] or num_cols)
            sz    = c3.selectbox("Size:", [c for c in num_cols if c not in [x, y]] or num_cols)
            color = c4.selectbox("Color:", ["None"] + cat_cols)
            # bubble size must be positive
            size_vals = df[sz].fillna(0).clip(lower=0)
            if size_vals.max() == 0:
                st.caption(f"⚠️ Size column `{sz}` is all zeros — defaulting to uniform size.")
                fig = px.scatter(df, x=x, y=y,
                                 color=None if color == "None" else color,
                                 opacity=0.7)
            else:
                df_b = df.copy()
                df_b[sz] = size_vals
                fig = px.scatter(df_b, x=x, y=y, size=sz, size_max=40,
                                 color=None if color == "None" else color,
                                 color_discrete_sequence=px.colors.qualitative.Set2,
                                 opacity=0.7)
            fig = fig_defaults(fig, f"Bubble: {y} vs {x}, size={sz}", height=500)

        elif "Violin" in chart_type:
            c1, c2 = st.columns(2)
            y_col = c1.selectbox("Numeric:", num_cols)
            x_col = c2.selectbox("Group:", ["None"] + cat_cols)
            fig = px.violin(df, y=y_col,
                            x=None if x_col == "None" else x_col,
                            color=None if x_col == "None" else x_col,
                            box=True, points="outliers",
                            color_discrete_sequence=px.colors.qualitative.Set2)
            fig = fig_defaults(fig, f"{y_col} Violin Plot")

        elif "Sunburst" in chart_type:
            c1, c2, c3 = st.columns(3)
            p   = c1.selectbox("Parent level:", cat_cols)
            remaining = [c for c in cat_cols if c != p]
            child = c2.selectbox("Child level:", remaining)
            val  = c3.selectbox("Value:", ["count"] + num_cols)
            if val == "count":
                df_sb = df.groupby([p, child]).size().reset_index(name="count")
                fig = px.sunburst(df_sb, path=[p, child], values="count",
                                  color_discrete_sequence=px.colors.qualitative.Set2)
            else:
                df_sb = df.groupby([p, child])[val].mean().reset_index()
                if df_sb[val].isnull().any() or (df_sb[val] <= 0).any():
                    df_sb[val] = df_sb[val].fillna(0).clip(lower=0.001)
                fig = px.sunburst(df_sb, path=[p, child], values=val,
                                  color_discrete_sequence=px.colors.qualitative.Set2)
            fig = fig_defaults(fig, "Sunburst Chart", height=500)

        elif "Area" in chart_type:
            c1, c2 = st.columns(2)
            x = c1.selectbox("Time:", date_cols)
            y = c2.selectbox("Value:", num_cols)
            fig = px.area(df.sort_values(x), x=x, y=y,
                          color_discrete_sequence=["#3b82f6"])
            fig = fig_defaults(fig, f"{y} — Area Chart")

        if fig:
            st.plotly_chart(fig, use_container_width=True)

    # ── TAB 3: CODE + NOTEBOOK DOWNLOAD ───────────────────────────────────────
    with tab3:
        st.markdown("### Visualization Code Gallery")

        col_code, col_nb = st.columns([3, 1])
        with col_nb:
            st.markdown("#### 📓 Download Notebook")
            st.markdown("Get all visualization code as a runnable Jupyter notebook:")

            nb_bytes = make_notebook(
                "Data Visualization — DS Workshop 2026",
                TOPIC_NOTEBOOKS["Visualization"]
            )
            st.download_button(
                label="⬇ Download .ipynb",
                data=nb_bytes,
                file_name="07_data_visualization.ipynb",
                mime="application/json",
                use_container_width=True,
            )

            st.markdown("---")
            st.markdown("#### 📦 All Topics")
            st.markdown("Download notebooks for every topic:")
            all_sections = []
            for topic, sections in TOPIC_NOTEBOOKS.items():
                all_sections.append({"heading": topic, "markdown": ""})
                all_sections.extend(sections)
            all_nb = make_notebook("DS Workshop 2026 — Complete Notebook", all_sections)
            st.download_button(
                label="⬇ Full Workshop .ipynb",
                data=all_nb,
                file_name="ds_workshop_2026_complete.ipynb",
                mime="application/json",
                use_container_width=True,
            )

        with col_code:
            chart_code = st.selectbox("Show code for:", [
                "Histogram", "Box Plot", "Scatter", "Heatmap",
                "Time Series", "Area Chart", "Violin", "Dashboard (subplots)"
            ])

            codes = {
                "Histogram": """import plotly.express as px

fig = px.histogram(
    df, x="temperature",
    nbins=50,
    color="material",
    barmode="overlay",
    opacity=0.8,
    title="Temperature Distribution by Material",
    labels={"temperature": "Temperature (°C)"},
    color_discrete_sequence=px.colors.qualitative.Set2,
)
fig.update_layout(template="plotly_dark", height=400)
fig.show()""",
                "Box Plot": """import plotly.express as px

fig = px.box(
    df, x="material", y="thickness_mm",
    color="material",
    points="outliers",
    notched=True,
    title="Thickness by Material — with Outliers",
    color_discrete_sequence=px.colors.qualitative.Set2,
)
fig.update_layout(template="plotly_dark")
fig.show()""",
                "Scatter": """import plotly.express as px

fig = px.scatter(
    df, x="temperature", y="vibration",
    color="severity",
    size="fuel_flow_kg_hr",
    hover_data=["engine_id", "timestamp"],
    opacity=0.7,
    title="Vibration vs Temperature",
)
fig.update_layout(template="plotly_dark", height=500)
fig.show()""",
                "Heatmap": """import plotly.graph_objects as go
import numpy as np

corr = df.select_dtypes(include=np.number).corr()

fig = go.Figure(go.Heatmap(
    z=corr.values,
    x=corr.columns,
    y=corr.index,
    colorscale="RdBu",
    zmid=0,
    text=corr.round(2).values,
    texttemplate="%{text}",
    showscale=True,
))
fig.update_layout(title="Correlation Matrix", template="plotly_dark")
fig.show()""",
                "Time Series": """import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                    subplot_titles=["Temperature", "Vibration", "Alerts"])

fig.add_trace(go.Scatter(x=df["timestamp"], y=df["egt_celsius"],
    name="Temp", line=dict(color="#ef4444", width=1.5)), row=1, col=1)

fig.add_trace(go.Scatter(x=df["timestamp"], y=df["vibration"],
    name="Vibration", line=dict(color="#f59e0b", width=1.5)), row=2, col=1)

fig.add_trace(go.Bar(x=df["timestamp"], y=df["alert_active"].astype(int),
    name="Alert", marker_color="#3b82f6"), row=3, col=1)

fig.update_layout(template="plotly_dark", height=600,
    title="Engine Telemetry Dashboard")
fig.show()""",
                "Area Chart": """import plotly.express as px

fig = px.area(
    df.sort_values("timestamp"),
    x="timestamp", y="temperature_C",
    color_discrete_sequence=["#3b82f6"],
    title="Temperature Over Time — Area Chart",
)
fig.update_layout(template="plotly_dark")
fig.show()""",
                "Violin": """import plotly.express as px

fig = px.violin(
    df, x="material", y="thickness_mm",
    color="material",
    box=True,
    points="outliers",
    title="Thickness Distribution by Material",
    color_discrete_sequence=px.colors.qualitative.Set2,
)
fig.update_layout(template="plotly_dark")
fig.show()""",
                "Dashboard (subplots)": """from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=["Thickness Dist", "By Material",
                    "Scatter", "Pass Rate"],
)

for mat in df["material"].unique():
    d = df[df["material"]==mat]["thickness_mm"]
    fig.add_trace(go.Histogram(x=d, name=mat, opacity=0.7), row=1, col=1)

fig.add_trace(go.Box(x=df["material"], y=df["thickness_mm"]), row=1, col=2)

fig.add_trace(go.Scatter(x=df["thickness_mm"], y=df["hardness_HRC"],
    mode="markers", marker=dict(opacity=0.5)), row=2, col=1)

rate = df.groupby("material")["passed_qc"].mean().reset_index()
fig.add_trace(go.Bar(x=rate["material"], y=rate["passed_qc"],
    marker_color="#10b981"), row=2, col=2)

fig.update_layout(template="plotly_dark", height=700,
    title="Manufacturing Dashboard")
fig.show()""",
            }
            st.code(codes[chart_code], language="python")

    # ── TAB 4: QUIZ ───────────────────────────────────────────────────────────
    with tab4:
        st.markdown("### Quiz — Data Visualization")
        questions = [
            {"q": "You want to show how salary varies across 5 departments. Best chart?",
             "options": ["Pie chart", "Box plot", "Line chart", "Area chart"],
             "answer": "Box plot",
             "explanation": "Box plots show distribution, median, quartiles, and outliers per group — perfect for comparing numerical distributions across categories."},
            {"q": "Which color scale is appropriate for a correlation matrix (values from -1 to +1)?",
             "options": ["Sequential (e.g. Blues)", "Diverging (e.g. RdBu)", "Rainbow", "Monochrome"],
             "answer": "Diverging (e.g. RdBu)",
             "explanation": "Diverging scales have a neutral midpoint (0 correlation) and colors diverge toward negative and positive — matching the data range perfectly."},
            {"q": "Your manager wants a pie chart for 12 product categories. What do you do?",
             "options": ["Use the pie chart", "Use a bar chart instead", "Use a 3D pie chart", "Use a radar chart"],
             "answer": "Use a bar chart instead",
             "explanation": "Pie charts work for ≤5 categories with clear dominance. 12 slices are unreadable. Horizontal bar charts are far more legible."},
            {"q": "A Line chart requires which column type that a Bar chart does NOT?",
             "options": ["Numeric column", "Categorical column", "Datetime column", "Boolean column"],
             "answer": "Datetime column",
             "explanation": "Line charts show trends OVER TIME — the X axis must be a datetime. Bar charts group by categorical columns instead."},
        ]
        for i, q in enumerate(questions):
            st.markdown(f"**Q{i+1}.** {q['q']}")
            choice = st.radio("", q["options"], key=f"quiz_07_q{i}", index=None, horizontal=True)
            if choice:
                if choice == q["answer"]:
                    st.markdown(f'<div class="box-success">✅ Correct! {q["explanation"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="box-warning">❌ Answer: **{q["answer"]}**. {q["explanation"]}</div>', unsafe_allow_html=True)
            st.markdown("")
        if "visualization" not in st.session_state.completed_topics:
            if st.button("✅ Mark topic as complete"):
                st.session_state.completed_topics.append("visualization")
                st.success("Topic marked complete!")
