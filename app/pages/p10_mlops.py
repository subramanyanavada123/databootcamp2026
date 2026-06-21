import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from app.utils.visualizations import fig_defaults
import plotly.graph_objects as go
import plotly.express as px

def render():
    st.markdown("""
    <div class="hero-banner" style="padding:1.5rem;">
      <h1 style="font-size:1.8rem;">🚀 MLOps & Production Systems</h1>
      <p>Shipping ML models to production — CI/CD, monitoring, drift detection, and retraining</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📖 Explain", "🎨 MLOps Simulator", "💻 Code", "✅ Quiz"])

    with tab1:
        st.markdown("### What is MLOps?")
        st.markdown("""
MLOps = DevOps + ML.

It's the discipline of **deploying, monitoring, and maintaining ML models in production**
with the same rigor we apply to software systems — versioning, testing, CI/CD, and observability.

> Without MLOps, ML models are science experiments.
> With MLOps, they're engineering products.
        """)

        st.code("""
THE MLOPS LIFECYCLE
─────────────────────────────────────────────────────────────────────
  Data     →  Model    →  Deploy  →  Monitor  →  Retrain  →  Loop
  ─────────────────────────────────────────────────────────────────
  Versioned   Tracked     API        Metrics     Trigger     CI/CD
  Validated   Registered  Container  Drift       Schedule    Auto
  Tested      Tested      A/B test   Alerts      Active      Gate

TOOLS
─────────────────────────────────────────────────────────────────────
  Experiment tracking:  MLflow, Weights & Biases
  Feature store:        Feast, Tecton, Vertex AI
  Model registry:       MLflow, SageMaker, Azure ML
  Serving:              FastAPI, BentoML, Triton, Ray Serve
  Monitoring:           Evidently, WhyLogs, Prometheus
  Orchestration:        Airflow, Prefect, ZenML
  CI/CD for ML:         GitHub Actions + pytest + DVC
        """, language="text")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Data Drift vs Model Drift")
            st.markdown("""
**Data Drift:** The input distribution shifts.
- Temperature sensor starts reading in Fahrenheit instead of Celsius
- New products introduced → different purchase patterns

**Concept Drift:** The relationship between features and target changes.
- COVID changed what "normal" customer behaviour looks like
- New competitors → price elasticity shifts

**Model Decay:** Performance degrades over time.
- Retrain trigger: AUC drops below threshold
- Schedule: Monthly / quarterly retraining
            """)

        with col2:
            st.markdown("### Deployment Strategies")
            deploy = {
                "Strategy": ["Blue-Green", "Canary", "Shadow", "A/B Test", "Champion/Challenger"],
                "How": ["2 envs, instant switch", "Gradual % traffic shift", "Parallel, no impact", "Split traffic, measure", "New vs old model"],
                "Risk": ["Low", "Medium", "None", "Controlled", "Controlled"],
                "Use when": ["Safety critical", "Gradual rollout", "Test in prod", "Model comparison", "Iterative improvement"],
            }
            st.dataframe(pd.DataFrame(deploy), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("""
        <div class="box-info">
        <strong>🧠 The ML in Production Truth:</strong><br>
        • 87% of ML projects never reach production (Gartner)<br>
        • Models degrade silently — average lifespan 3-6 months without monitoring<br>
        • The #1 MLOps mistake: deploying once, never monitoring<br>
        • Best practice: automate retraining triggers based on drift metrics
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("### MLOps Simulator")

        sim_type = st.selectbox("Scenario:", [
            "📉 Model Performance Monitoring",
            "🌊 Data Drift Detection",
            "🔄 A/B Model Comparison",
            "🏭 Deployment Architecture",
        ])

        np.random.seed(42)

        if "Monitoring" in sim_type:
            st.markdown("### Model Performance Over Time")
            days = pd.date_range("2026-01-01", periods=180, freq="D")
            perf = 0.92 - np.cumsum(np.random.normal(0.0005, 0.003, 180))
            perf = np.clip(perf, 0.55, 0.95)

            threshold = st.slider("Alert threshold (AUC):", 0.7, 0.9, 0.82)
            retrain_day = st.slider("Retrain at day:", 60, 150, 100)

            perf[retrain_day:] = np.clip(0.91 - np.cumsum(np.random.normal(0.0005, 0.002, 180 - retrain_day)), 0.75, 0.95)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=days, y=perf, name="Model AUC",
                                     line=dict(color="#3b82f6", width=2)))
            fig.add_hline(y=threshold, line_dash="dash", line_color="#ef4444",
                          annotation_text=f"Alert threshold: {threshold}")
            fig.add_vline(x=days[retrain_day], line_dash="dot", line_color="#10b981",
                          annotation_text="Retrain")
            breach = days[np.where(perf < threshold)[0][0]] if (perf < threshold).any() else None
            if breach:
                fig.add_vline(x=breach, line_dash="dash", line_color="#f59e0b",
                              annotation_text="⚠ Alert triggered")
            fig = fig_defaults(fig, "Production Model AUC — 6 Month Monitor", height=400)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"**Model breached threshold at day {np.where(perf < threshold)[0][0] if (perf < threshold).any() else 'N/A'}**")

        elif "Drift" in sim_type:
            st.markdown("### Data Drift Detection")
            st.markdown("Comparing feature distributions between training data and recent production data.")

            feature = st.selectbox("Feature to monitor:", ["temperature_C", "vibration", "fuel_flow_kg_hr"])
            drift_amount = st.slider("Inject drift (shift in std devs):", 0.0, 3.0, 1.5)

            train_data = np.random.normal(0, 1, 1000)
            prod_data  = np.random.normal(drift_amount * 0.5, 1 + drift_amount * 0.2, 500)

            fig = go.Figure()
            fig.add_trace(go.Histogram(x=train_data, name="Training", opacity=0.7,
                                       marker_color="#3b82f6", nbinsx=40))
            fig.add_trace(go.Histogram(x=prod_data, name="Production", opacity=0.7,
                                       marker_color="#ef4444", nbinsx=40))
            fig.update_layout(barmode="overlay")
            fig = fig_defaults(fig, f"Distribution Drift: {feature}", height=400)
            st.plotly_chart(fig, use_container_width=True)

            from scipy import stats
            ks_stat, ks_p = stats.ks_2samp(train_data, prod_data)
            drift_detected = ks_p < 0.05

            st.metric("KS Statistic", f"{ks_stat:.3f}")
            st.metric("P-value", f"{ks_p:.4f}")
            if drift_detected:
                st.markdown('<div class="box-warning">⚠️ Drift DETECTED (p < 0.05) — Consider retraining!</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="box-success">✅ No significant drift detected.</div>', unsafe_allow_html=True)

        elif "A/B" in sim_type:
            st.markdown("### A/B Model Comparison")
            n_requests = st.slider("Total requests:", 100, 5000, 1000)
            split = st.slider("Traffic to new model (%):", 10, 50, 20)

            n_a = int(n_requests * (1 - split/100))
            n_b = int(n_requests * split/100)

            model_a_acc = np.random.binomial(n_a, 0.85) / n_a
            model_b_acc = np.random.binomial(n_b, 0.88) / n_b

            days_ab = pd.date_range("2026-06-01", periods=30, freq="D")
            perf_a = np.random.normal(0.85, 0.02, 30)
            perf_b = np.random.normal(0.88, 0.025, 30)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=days_ab, y=perf_a, name="Champion (A)", line=dict(color="#3b82f6", width=2)))
            fig.add_trace(go.Scatter(x=days_ab, y=perf_b, name="Challenger (B)", line=dict(color="#10b981", width=2)))
            fig = fig_defaults(fig, "A/B Model Test — Daily Accuracy", height=350)
            st.plotly_chart(fig, use_container_width=True)

            col1, col2, col3 = st.columns(3)
            col1.metric("Model A (Champion)", f"{model_a_acc*100:.1f}%", f"{n_a} requests")
            col2.metric("Model B (Challenger)", f"{model_b_acc*100:.1f}%", f"{n_b} requests")
            col3.metric("Improvement", f"{(model_b_acc - model_a_acc)*100:+.1f}%",
                        "🟢 Promote B" if model_b_acc > model_a_acc + 0.01 else "⚠ Not significant")

        else:
            st.markdown("### MLOps Deployment Architecture")
            st.markdown("""
```
┌─────────────────────────────────────────────────────────────────┐
│                    PRODUCTION ML SYSTEM                         │
│                                                                 │
│  Data Sources          Feature Store         Model Registry     │
│  ─────────────         ─────────────         ─────────────     │
│  Sensors ──────►  ETL  ► Online Store ──────► Champion v2.3    │
│  APIs ──────────►      ► Offline Store       ► Challenger v2.4  │
│  Events ────────►                                               │
│         │                                          │            │
│         ▼                                          ▼            │
│  Data Validation                          Model Serving         │
│  (Great Expectations)                     (FastAPI / Triton)    │
│         │                                          │            │
│         ▼                                          ▼            │
│  Feature Pipeline                         Monitoring            │
│  (Airflow / Prefect)                      (Evidently / Prom)   │
│                                                    │            │
│                              Drift Alert ◄─────────┘           │
│                                   │                             │
│                                   ▼                             │
│                            Auto-Retrain                         │
│                            (CI/CD Pipeline)                     │
└─────────────────────────────────────────────────────────────────┘
```
            """)

    with tab3:
        st.markdown("### MLOps Code")
        example = st.selectbox("Example:", [
            "FastAPI Model Serving",
            "MLflow Experiment Tracking",
            "Data Drift Detection (Evidently)",
            "Automated Retraining Pipeline",
        ])

        codes = {
            "FastAPI Model Serving": """
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import logging

app = FastAPI(title="Churn Prediction API", version="2.3")
model = joblib.load("models/churn_model_v2.3.pkl")
logger = logging.getLogger(__name__)

class PredictRequest(BaseModel):
    tenure_months: int
    monthly_charges: float
    contract_type: str
    internet_service: str
    num_complaints: int

class PredictResponse(BaseModel):
    churn_probability: float
    will_churn: bool
    confidence: str
    model_version: str = "2.3"

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        df = pd.DataFrame([req.dict()])
        prob = model.predict_proba(df)[0, 1]
        return PredictResponse(
            churn_probability=round(float(prob), 4),
            will_churn=prob > 0.5,
            confidence="high" if abs(prob - 0.5) > 0.3 else "medium",
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "healthy", "model": "churn_v2.3"}

# Run: uvicorn main:app --host 0.0.0.0 --port 8000
            """,
            "MLflow Experiment Tracking": """
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

mlflow.set_tracking_uri("http://mlflow.internal:5000")
mlflow.set_experiment("churn-prediction-v2")

df = pd.read_parquet("train_data.parquet")
X = df.drop("churned", axis=1)
y = df["churned"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Grid search with MLflow tracking
for n_est in [50, 100, 200]:
    for depth in [5, 10, None]:
        with mlflow.start_run(run_name=f"rf_n{n_est}_d{depth}"):
            model = RandomForestClassifier(n_estimators=n_est, max_depth=depth)
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
            acc = accuracy_score(y_test, model.predict(X_test))

            # Log parameters
            mlflow.log_params({"n_estimators": n_est, "max_depth": depth})
            # Log metrics
            mlflow.log_metrics({"auc": auc, "accuracy": acc})
            # Log model
            mlflow.sklearn.log_model(model, "model",
                registered_model_name="churn-predictor")
            print(f"n={n_est}, depth={depth}: AUC={auc:.3f}")
            """,
            "Data Drift Detection (Evidently)": """
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import *

# Reference (training) vs Current (production)
reference = pd.read_parquet("reference_data.parquet")
current   = pd.read_parquet("production_data_last_7days.parquet")

# ── Drift Report ──────────────────────────────────────────────────
drift_report = Report(metrics=[
    DataDriftPreset(),
    DataQualityPreset(),
    ColumnDriftMetric(column_name="temperature"),
    ColumnDriftMetric(column_name="vibration"),
    DatasetMissingValuesMetric(),
])
drift_report.run(reference_data=reference, current_data=current)

# Save HTML report
drift_report.save_html("reports/drift_report.html")

# Get programmatic results for alerting
result = drift_report.as_dict()
n_drifted = result["metrics"][0]["result"]["number_of_drifted_columns"]
drift_share = result["metrics"][0]["result"]["share_of_drifted_columns"]

if drift_share > 0.3:
    print(f"⚠️  ALERT: {n_drifted} features drifted ({drift_share:.0%})")
    # Trigger retraining pipeline
    trigger_retraining()
            """,
            "Automated Retraining Pipeline": """
# Airflow DAG for scheduled model retraining
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import joblib

default_args = {
    "owner": "mlops-team",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": True,
    "email": ["mlops@company.com"],
}

def extract_training_data(**ctx):
    # Pull last 90 days of production data
    end   = ctx["execution_date"]
    start = end - timedelta(days=90)
    df = load_from_warehouse(start, end)
    df.to_parquet(f"/tmp/train_{end.date()}.parquet")

def validate_data(**ctx):
    df = pd.read_parquet(f"/tmp/train_{ctx['execution_date'].date()}.parquet")
    assert len(df) > 5000,    "Not enough training data"
    assert df.isnull().mean().max() < 0.1, "Too many missing values"
    assert df["churned"].mean() > 0.05,    "Target too sparse"

def retrain_model(**ctx):
    df = pd.read_parquet(f"/tmp/train_{ctx['execution_date'].date()}.parquet")
    # ... training code ...
    model = train_pipeline(df)
    joblib.dump(model, f"/models/churn_model_{ctx['execution_date'].date()}.pkl")

def evaluate_and_promote(**ctx):
    new_model  = joblib.load(f"/models/churn_model_{ctx['execution_date'].date()}.pkl")
    curr_model = joblib.load("/models/champion.pkl")
    X_test, y_test = load_holdout()
    new_auc  = roc_auc_score(y_test, new_model.predict_proba(X_test)[:, 1])
    curr_auc = roc_auc_score(y_test, curr_model.predict_proba(X_test)[:, 1])
    if new_auc > curr_auc + 0.01:
        joblib.dump(new_model, "/models/champion.pkl")
        print(f"✅ New model promoted! AUC: {new_auc:.3f} vs {curr_auc:.3f}")
    else:
        print(f"⚠️ Keeping champion. New: {new_auc:.3f} vs Current: {curr_auc:.3f}")

with DAG("ml-retraining", schedule_interval="@weekly",
         start_date=datetime(2026, 1, 1), default_args=default_args) as dag:
    t1 = PythonOperator(task_id="extract",   python_callable=extract_training_data)
    t2 = PythonOperator(task_id="validate",  python_callable=validate_data)
    t3 = PythonOperator(task_id="retrain",   python_callable=retrain_model)
    t4 = PythonOperator(task_id="promote",   python_callable=evaluate_and_promote)
    t1 >> t2 >> t3 >> t4
            """,
        }
        st.code(codes[example], language="python")

    with tab4:
        st.markdown("### Quiz — MLOps")
        questions = [
            {"q": "You deploy a model in January. By June, accuracy has dropped from 91% to 73%. What's most likely?",
             "options": ["Bug in the code", "Concept or data drift", "Server overload", "Wrong evaluation metric"],
             "answer": "Concept or data drift",
             "explanation": "Silent model decay over months is the classic symptom of drift — the world changed but your model didn't."},
            {"q": "What is a Champion/Challenger deployment?",
             "options": ["Deploy only the best model", "Run two models in parallel, compare, promote the better one", "A/B test users not models", "Train twice for robustness"],
             "answer": "Run two models in parallel, compare, promote the better one",
             "explanation": "The champion serves most traffic. A challenger receives a small % to prove itself. If it wins, it becomes the new champion."},
            {"q": "Which tool tracks ML experiments (params, metrics, artifacts)?",
             "options": ["Airflow", "Docker", "MLflow", "Kafka"],
             "answer": "MLflow",
             "explanation": "MLflow tracks experiment runs, logs parameters, metrics and artifacts, and provides a model registry for versioning."},
        ]
        for i, q in enumerate(questions):
            st.markdown(f"**Q{i+1}.** {q['q']}")
            choice = st.radio("", q["options"], key=f"quiz_10_q{i}", index=None, horizontal=True)
            if choice:
                if choice == q["answer"]:
                    st.markdown(f'<div class="box-success">✅ Correct! {q["explanation"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="box-warning">❌ Answer: **{q["answer"]}**. {q["explanation"]}</div>', unsafe_allow_html=True)
            st.markdown("")
        if "mlops" not in st.session_state.completed_topics:
            if st.button("✅ Mark topic as complete"):
                st.session_state.completed_topics.append("mlops")
                st.success("Topic marked complete!")
