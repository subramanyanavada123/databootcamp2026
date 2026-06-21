import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from app.utils.datasets import DATASETS

CHALLENGES = [
    {
        "id": "churn",
        "title": "Telecom Churn Predictor",
        "industry": "📱 Telecom",
        "difficulty": "⭐⭐ Medium",
        "time": "45 min",
        "problem": """A telecom company is losing 27% of customers monthly.
Your task is to build a model that identifies which customers are at risk of churning
**before** they leave, so the retention team can intervene.

**Business goal:** Reduce churn rate from 27% → below 15% by targeting high-risk customers.""",
        "dataset_key": "📱 Telecom Churn (Telco)",
        "tasks": [
            "Load the dataset and perform initial EDA — understand the churn rate, distributions, and missing values",
            "Clean the data: handle missing values in avg_call_duration",
            "Feature engineering: create 'charges_per_month_tenure' = total_charges / tenure_months",
            "Encode categorical features (contract_type, internet_service)",
            "Train a Random Forest classifier to predict 'churned'",
            "Evaluate with ROC AUC — target > 0.75",
            "Find top 3 most important features for churn",
            "BONUS: Identify the 10 highest-risk customers in the dataset",
        ],
        "hints": [
            "Use df.isnull().sum() to find missing values first",
            "pd.get_dummies() for one-hot encoding",
            "model.feature_importances_ after fitting RandomForestClassifier",
            "model.predict_proba(X)[:, 1] gives churn probability per customer",
        ],
        "expected": "ROC AUC > 0.75, feature importance chart, list of top 10 at-risk customers"
    },
    {
        "id": "anomaly",
        "title": "Factory Sensor Anomaly Detector",
        "industry": "🔩 Manufacturing",
        "difficulty": "⭐⭐ Medium",
        "time": "45 min",
        "problem": """A factory floor manager wants an automated system to flag sensor readings
that indicate potential equipment failure — BEFORE the machine breaks down.

**Business goal:** Reduce unplanned downtime by 40% through early anomaly detection.""",
        "dataset_key": "✈️ Aircraft Engine (Aerospace)",
        "tasks": [
            "Load the engine dataset and explore distributions of EGT, vibration, and N1 speed",
            "Clean missing values using forward-fill (appropriate for time series sensor data)",
            "Visualize the time series for all 3 sensor readings",
            "Use IQR method to identify outliers in vibration and EGT",
            "Train an Isolation Forest on [egt_celsius, vibration, n1_fan_speed]",
            "Add anomaly score and is_anomaly columns to the DataFrame",
            "BONUS: Calculate what percentage of readings are anomalies — does it match alert_active?",
        ],
        "hints": [
            "df.fillna(method='ffill') for time series",
            "IsolationForest(contamination=0.05) — assume 5% anomaly rate",
            "model.fit_predict(X) returns 1 (normal) or -1 (anomaly)",
            "Compare is_anomaly with alert_active using df.groupby(['is_anomaly', 'alert_active']).size()",
        ],
        "expected": "Anomaly scores, time series with flagged anomalies highlighted, confusion vs alert_active"
    },
    {
        "id": "iot",
        "title": "Smart Building Energy Optimizer",
        "industry": "🌡️ IoT",
        "difficulty": "⭐⭐⭐ Hard",
        "time": "60 min",
        "problem": """A smart building wants to predict when energy usage will be highest
so HVAC systems can pre-cool before peak hours, reducing energy costs by 20%.

**Business goal:** Forecast temperature 1 hour ahead with < 1.5°C MAE.""",
        "dataset_key": "🌡️ IoT Sensors (Smart Building)",
        "tasks": [
            "Load the IoT dataset and explore the temperature time series",
            "Extract datetime features: hour, day_of_week, is_weekend from timestamp",
            "Create lag features: temperature at t-1, t-2, t-3 (previous readings)",
            "Create rolling features: 5-period rolling mean and std",
            "Create a target: temperature 12 steps ahead (1 hour if 5min intervals)",
            "Split data chronologically (first 80% train, last 20% test — no random shuffle!)",
            "Train a GradientBoostingRegressor",
            "Evaluate with MAE and RMSE — target MAE < 1.5°C",
            "BONUS: Visualize actual vs predicted temperature on the test set",
        ],
        "hints": [
            "df.sort_values('timestamp') before creating lags",
            "df['temp_lag_1'] = df['temperature_C'].shift(1)",
            "df['target'] = df['temperature_C'].shift(-12)  # 12 steps ahead",
            "Split by index: split = int(len(df) * 0.8); train=df[:split]; test=df[split:]",
            "from sklearn.metrics import mean_absolute_error",
        ],
        "expected": "MAE < 1.5°C, actual vs predicted plot, feature importance for lag features"
    },
]

def render():
    st.markdown("""
    <div class="hero-banner" style="padding:1.5rem;">
      <h1 style="font-size:1.8rem;">⚡ Challenge Mode</h1>
      <p>Hands-on mini-projects — real datasets, real problems, measurable outcomes</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Pick Your Challenge")
    cols = st.columns(len(CHALLENGES))
    for i, ch in enumerate(CHALLENGES):
        with cols[i]:
            st.markdown(f"""
            <div class="ds-card" style="text-align:center; cursor:pointer;">
              <div style="font-size:1.5rem; margin-bottom:0.3rem;">{ch['industry'].split()[0]}</div>
              <div style="font-weight:700; color:#f1f5f9; font-size:0.9rem;">{ch['title']}</div>
              <div style="font-size:0.75rem; color:#64748b; margin-top:0.3rem;">{ch['difficulty']} · {ch['time']}</div>
            </div>
            """, unsafe_allow_html=True)

    challenge_titles = [f"{c['industry']} — {c['title']}" for c in CHALLENGES]
    selected = st.selectbox("Select challenge:", challenge_titles)
    ch = CHALLENGES[challenge_titles.index(selected)]

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.markdown(f"**Industry:** {ch['industry']}")
    col2.markdown(f"**Difficulty:** {ch['difficulty']}")
    col3.markdown(f"**Time:** {ch['time']}")

    st.markdown("### Problem Statement")
    st.markdown(ch["problem"])

    st.markdown("### Dataset")
    df = DATASETS[ch["dataset_key"]]()
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{len(df):,}")
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing", df.isnull().sum().sum())
    st.dataframe(df.head(10), use_container_width=True)
    st.download_button("⬇ Download Dataset", df.to_csv(index=False),
                       file_name=f"challenge_{ch['id']}.csv", mime="text/csv")

    st.markdown("### Your Tasks")
    for i, task in enumerate(ch["tasks"]):
        is_bonus = "BONUS" in task
        prefix = "🎯 BONUS:" if is_bonus else f"**{i+1}.**"
        task_text = task.replace("BONUS: ", "")
        if is_bonus:
            st.markdown(f'<div class="box-info">{prefix} {task_text}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f"{prefix} {task_text}")

    hints_expanded = st.expander("💡 Show Hints", expanded=False)
    with hints_expanded:
        for hint in ch["hints"]:
            st.markdown(f"- `{hint}`")

    st.markdown("### Expected Output")
    st.markdown(f'<div class="box-success">✅ {ch["expected"]}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Your Workspace")
    st.markdown("Write and run your solution below:")

    starter_code = f"""import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.impute import SimpleImputer
import plotly.express as px

# Load the dataset
# (In the workshop, df is already loaded above — use the Download button)
# df = pd.read_csv('challenge_{ch['id']}.csv')

# Step 1: EDA
print(df.shape)
print(df.isnull().sum())
print(df.describe())

# Continue your solution here...
"""
    user_code = st.text_area("Your code:", value=starter_code, height=350)

    if st.button("▶ Run Code", type="primary"):
        with st.spinner("Running..."):
            try:
                local_ns = {"df": df}
                exec(user_code, {"pd": pd, "np": np, "st": st,
                                  "__builtins__": __builtins__}, local_ns)
                st.success("✅ Code ran successfully!")
            except Exception as e:
                st.error(f"❌ Error: {e}")

    st.markdown("---")
    solution_exp = st.expander("🔑 View Solution (try yourself first!)", expanded=False)
    with solution_exp:
        if ch["id"] == "churn":
            st.code("""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer

# 1. EDA
print(df.shape, df['churned'].mean().round(3))
print(df.isnull().sum())

# 2. Clean
df['avg_call_duration'].fillna(df['avg_call_duration'].median(), inplace=True)

# 3. Feature engineering
df['charges_per_month'] = df['total_charges'] / df['tenure_months'].replace(0, 1)

# 4. Encode
df = pd.get_dummies(df, columns=['contract_type', 'internet_service'], drop_first=True)

# 5. Prepare
features = [c for c in df.select_dtypes(include=np.number).columns
            if c not in ['churned']]
X = df[features]
y = df['churned'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                      stratify=y, random_state=42)
imp = SimpleImputer(strategy='median')
X_train = imp.fit_transform(X_train)
X_test  = imp.transform(X_test)

# 6. Train & evaluate
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print(f"ROC AUC: {auc:.3f}")

# 7. Feature importance
fi = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print(fi.head(3))

# BONUS: Top 10 at-risk customers
df_test = df.copy()
X_all = imp.transform(df[features])
df_test['churn_prob'] = model.predict_proba(X_all)[:, 1]
top10 = df_test.nlargest(10, 'churn_prob')[['customer_id', 'churn_prob', 'monthly_charges']]
print(top10)
            """, language="python")
