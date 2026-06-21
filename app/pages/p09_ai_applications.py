import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from app.utils.visualizations import fig_defaults, plot_pca_2d
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Shared style helper ────────────────────────────────────────────────────────
def info(text):
    st.markdown(f'<div class="box-info">{text}</div>', unsafe_allow_html=True)

def success(text):
    st.markdown(f'<div class="box-success">{text}</div>', unsafe_allow_html=True)

def warn(text):
    st.markdown(f'<div class="box-warning">{text}</div>', unsafe_allow_html=True)

def section(icon, title, body=""):
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:0.6rem;margin:1.5rem 0 0.5rem 0;
                padding-bottom:0.4rem;border-bottom:1px solid #334155;">
      <span style="font-size:1.3rem;">{icon}</span>
      <span style="font-weight:700;color:#f1f5f9;font-size:1.05rem;">{title}</span>
    </div>
    """, unsafe_allow_html=True)
    if body:
        st.markdown(body)

# ── Page ──────────────────────────────────────────────────────────────────────
def render():
    st.markdown("""
    <div class="hero-banner" style="padding:1.5rem;">
      <h1 style="font-size:1.8rem;">🧠 AI Applications</h1>
      <p>Deep-dive into every major AI technique — fully interactive, no API key needed</p>
    </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs([
        "📖 AI Landscape",
        "🔍 Anomaly Detection",
        "📊 Clustering",
        "🔮 Forecasting",
        "🧮 Dimensionality",
        "📝 NLP Pipeline",
        "🧬 Neural Networks",
        "🤖 LLMs & RAG",
        "👁️ Computer Vision",
        "✅ Quiz",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — AI LANDSCAPE
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[0]:
        st.markdown("### The Complete AI Landscape")
        st.code("""
ARTIFICIAL INTELLIGENCE  (making machines behave intelligently)
│
├── RULE-BASED AI          (if-then logic, expert systems, 1950s–80s)
│    └── Still used in safety-critical systems (avionics, medical devices)
│
├── MACHINE LEARNING       (learn patterns from data, 1990s–now)
│    │
│    ├── Classical ML      (features hand-crafted by humans)
│    │    ├── Supervised     → Random Forest, XGBoost, SVM, Logistic Reg
│    │    ├── Unsupervised   → K-Means, DBSCAN, Isolation Forest, PCA
│    │    └── Reinforcement  → Q-Learning, PPO (game AI, robotics)
│    │
│    └── DEEP LEARNING     (features learned automatically from raw data)
│         ├── CNN           → Images  (defect detection, medical imaging)
│         ├── RNN / LSTM    → Sequences  (time series, audio, sensor data)
│         ├── Transformer   → Language + Code + Vision  (BERT, GPT, ViT)
│         ├── Autoencoder   → Anomaly detection, compression, generation
│         ├── GAN           → Synthetic data, image generation
│         └── Diffusion     → Stable Diffusion, DALL-E, Sora
│
├── NATURAL LANGUAGE PROCESSING
│    ├── Classic NLP       → TF-IDF, Bag-of-Words, regex pipelines
│    ├── Word Embeddings   → Word2Vec, GloVe, FastText
│    ├── Pre-trained LLMs  → BERT (understanding), GPT (generation)
│    └── RAG Systems       → LLM + your private knowledge base
│
├── COMPUTER VISION
│    ├── Classification    → "Is this a crack?" (one label per image)
│    ├── Detection         → "Where are the cracks?" (bounding boxes)
│    ├── Segmentation      → "Which pixels are cracked?" (pixel masks)
│    └── Generation        → Synthetic defect images for training
│
└── GENERATIVE AI          (creates new content)
     ├── Text   → Claude, GPT-4, Llama, Mistral
     ├── Image  → Stable Diffusion, DALL-E, Midjourney
     ├── Code   → GitHub Copilot, Claude Code, Cursor
     ├── Audio  → ElevenLabs, Whisper (transcription)
     └── Video  → Sora, Runway, Pika
        """, language="text")

        col1, col2 = st.columns(2)
        with col1:
            section("🆚", "AI vs ML vs DL vs GenAI — What's the difference?")
            comparison = pd.DataFrame({
                "Term": ["Artificial Intelligence", "Machine Learning", "Deep Learning", "Generative AI"],
                "Scope": ["Broadest umbrella", "Subset of AI", "Subset of ML", "Subset of DL"],
                "How it works": ["Rules OR learning", "Stats + data patterns", "Multi-layer neural nets", "Learns data distribution"],
                "Data needed": ["None to millions", "Thousands–millions", "100K–billions", "Billions of tokens"],
                "Engineering example": ["Flight autopilot rules", "Churn predictor", "Weld defect CNN", "Maintenance chatbot"],
                "Interpretable?": ["Usually yes", "Varies", "Usually no", "No"],
            })
            st.dataframe(comparison, use_container_width=True, hide_index=True)

        with col2:
            section("🏭", "Industry AI Applications Matrix")
            apps = pd.DataFrame({
                "Industry": ["Aerospace", "Manufacturing", "Telecom", "Healthcare", "Finance", "Automotive", "Energy", "Retail"],
                "AI Application": ["Predictive maintenance", "Visual defect detection", "Network anomaly", "Medical diagnosis", "Fraud detection", "Autonomous driving", "Load forecasting", "Demand prediction"],
                "Data Type": ["Sensor time series", "Images + sensors", "Network logs", "Images + tabular", "Transactions", "Video + LIDAR", "Time series", "Sales history"],
                "Model": ["LSTM / XGBoost", "YOLOv8 / CNN", "Isolation Forest", "CNN / ViT", "Gradient Boosting", "CNN + RL", "Prophet / LSTM", "XGBoost / ARIMA"],
            })
            st.dataframe(apps, use_container_width=True, hide_index=True)

        section("📐", "The Data → AI Value Chain")
        st.code("""
RAW SIGNAL                                                    BUSINESS OUTCOME
──────────────────────────────────────────────────────────────────────────────
Vibration        Clean    Engineer    Train     Deploy    Predict     Save
sensor data  →  Remove →  lag &   →  LSTM   →  FastAPI →  bearing →  £2M/yr
(10kHz)         noise     rolling    model      on K8s     failure    downtime
                           stats                           72hr ahead
──────────────────────────────────────────────────────────────────────────────
    Stage 3    Stage 4   Stage 6   Stage 8   Stage 10   Stage 9
  (Cleaning) (Process) (Feat Eng)  (ML)     (MLOps)    (AI App)
        """, language="text")

        info("""
<strong>🧠 Key insight for engineers:</strong> AI is just <em>software</em> that learns its own parameters from data
instead of being hand-coded. A neural network with 175 billion parameters (GPT-3) is still just
matrix multiplications — billions of them — learned by gradient descent over trillions of text tokens.<br><br>
The engineering challenge is: <strong>getting the right data → in the right format → to the right model →
served reliably in production</strong>. That's what Stages 1–10 of this workshop are all about.
        """)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — ANOMALY DETECTION
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[1]:
        section("🔍", "Anomaly Detection — How Machines Find the Unexpected")
        st.markdown("""
**Anomaly detection** finds data points that are **significantly different** from the norm —
without needing labelled examples of what "bad" looks like.

This is critical in engineering because:
- You have **millions of normal readings** but only **dozens of failures**
- Labelling every failure is expensive and often impossible
- Anomalies in sensors often precede real failures by hours or days
        """)

        st.markdown("### Three Core Approaches")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
**Statistical**
- Z-score (>3σ = anomaly)
- IQR method
- Moving average deviation
- Assumes distribution
- Fast, interpretable
- ✅ Simple sensors
            """)
        with col2:
            st.markdown("""
**Machine Learning**
- Isolation Forest
- One-Class SVM
- DBSCAN
- No distribution assumption
- Handles multivariate
- ✅ Multi-sensor systems
            """)
        with col3:
            st.markdown("""
**Deep Learning**
- Autoencoder (reconstruct)
- LSTM prediction error
- VAE (variational)
- Learns complex patterns
- Needs more data
- ✅ Raw signals/images
            """)

        st.markdown("---")
        st.markdown("### Live Demo — Isolation Forest")

        info("""
<strong>How Isolation Forest works:</strong> Build many random decision trees.
Normal points need <em>many splits</em> to isolate (they're in dense clusters).
Anomalies need <em>few splits</em> (they're alone in sparse space).<br>
<strong>Anomaly score = average path length to isolation.</strong> Short path = anomaly.
        """)

        col1, col2 = st.columns([2, 1])
        with col2:
            st.markdown("#### Controls")
            scenario = st.selectbox("Scenario:", [
                "Manufacturing: Vibration + Temperature",
                "Finance: Transaction Amount + Frequency",
                "Network: Packet Size + Latency",
            ])
            contamination = st.slider("Expected anomaly %:", 1, 20, 5)
            n_normal = st.slider("Normal points:", 100, 500, 250)
            show_scores = st.checkbox("Show anomaly scores", value=True)

        with col1:
            np.random.seed(42)
            if "Manufacturing" in scenario:
                X_normal = np.random.multivariate_normal([640, 0.4], [[100, 0.01], [0.01, 0.02]], n_normal)
                X_anom   = np.vstack([
                    np.random.multivariate_normal([700, 2.5], [[25, 0], [0, 0.1]], 8),
                    np.random.multivariate_normal([580, 0.1], [[25, 0], [0, 0.005]], 7),
                ])
                xlab, ylab = "EGT (°C)", "Vibration (g)"
            elif "Finance" in scenario:
                X_normal = np.random.multivariate_normal([50, 3], [[200, 0.5], [0.5, 1]], n_normal)
                X_anom   = np.vstack([
                    np.random.multivariate_normal([500, 20], [[100, 0], [0, 5]], 10),
                    np.random.multivariate_normal([10, 0.1], [[5, 0], [0, 0.01]], 5),
                ])
                xlab, ylab = "Transaction Amount ($)", "Transactions/Hour"
            else:
                X_normal = np.random.multivariate_normal([1500, 5], [[10000, 0], [0, 1]], n_normal)
                X_anom   = np.vstack([
                    np.random.multivariate_normal([8000, 50], [[1000, 0], [0, 25]], 10),
                    np.random.multivariate_normal([50, 200], [[100, 0], [0, 100]], 5),
                ])
                xlab, ylab = "Packet Size (bytes)", "Latency (ms)"

            from sklearn.ensemble import IsolationForest
            X_all = np.vstack([X_normal, X_anom])
            true_labels = ["Normal"] * n_normal + ["True Anomaly"] * len(X_anom)

            model = IsolationForest(contamination=contamination/100, n_estimators=100, random_state=42)
            preds = model.fit_predict(X_all)
            scores = model.score_samples(X_all)   # more negative = more anomalous

            pred_labels = ["Detected Anomaly" if p == -1 else "Predicted Normal" for p in preds]

            df_plot = pd.DataFrame({xlab: X_all[:,0], ylab: X_all[:,1],
                                    "True Label": true_labels, "Prediction": pred_labels,
                                    "Anomaly Score": scores.round(3)})

            fig = px.scatter(df_plot, x=xlab, y=ylab,
                             color="Prediction", symbol="True Label",
                             opacity=0.8, hover_data=["Anomaly Score"],
                             color_discrete_map={"Detected Anomaly": "#ef4444", "Predicted Normal": "#3b82f6"})
            fig = fig_defaults(fig, "Isolation Forest — Anomaly Detection", height=420)
            st.plotly_chart(fig, use_container_width=True)

            # Confusion
            tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == "True Anomaly" and p == "Detected Anomaly")
            fp = sum(1 for t, p in zip(true_labels, pred_labels) if t == "Normal" and p == "Detected Anomaly")
            fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == "True Anomaly" and p == "Predicted Normal")
            prec = tp / max(tp + fp, 1)
            rec  = tp / max(tp + fn, 1)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Detected anomalies", (preds == -1).sum())
            c2.metric("True positives", tp)
            c3.metric("Precision", f"{prec:.0%}")
            c4.metric("Recall", f"{rec:.0%}")

        if show_scores:
            st.markdown("#### Anomaly Score Distribution")
            fig_sc = px.histogram(df_plot, x="Anomaly Score", color="True Label",
                                  nbins=40, barmode="overlay", opacity=0.8,
                                  color_discrete_sequence=["#3b82f6", "#ef4444"])
            fig_sc = fig_defaults(fig_sc, "Score distribution — more negative = more anomalous", height=280)
            st.plotly_chart(fig_sc, use_container_width=True)

        section("📊", "Statistical Approach — Z-Score Comparison")
        st.markdown("Z-score flags readings more than N standard deviations from the mean.")
        t = pd.date_range("2026-06-01", periods=200, freq="5min")
        signal = np.random.normal(640, 8, 200)
        signal[80:85]  += 60   # spike up
        signal[140:143] -= 40  # spike down
        z_thresh = st.slider("Z-score threshold (σ):", 1.5, 4.0, 2.5, step=0.25)
        z = np.abs((signal - signal.mean()) / signal.std())
        is_anom = z > z_thresh
        fig_z = go.Figure()
        fig_z.add_trace(go.Scatter(x=t, y=signal, name="EGT", line=dict(color="#3b82f6", width=1.5)))
        fig_z.add_trace(go.Scatter(x=t[is_anom], y=signal[is_anom], mode="markers",
                                   name="Z-score anomaly", marker=dict(color="#ef4444", size=10, symbol="x")))
        fig_z.add_hline(y=signal.mean() + z_thresh * signal.std(), line_dash="dash", line_color="#f59e0b",
                        annotation_text=f"+{z_thresh}σ")
        fig_z.add_hline(y=signal.mean() - z_thresh * signal.std(), line_dash="dash", line_color="#f59e0b",
                        annotation_text=f"-{z_thresh}σ")
        fig_z = fig_defaults(fig_z, f"Z-Score Anomaly Detection (threshold={z_thresh}σ)", height=320)
        st.plotly_chart(fig_z, use_container_width=True)
        st.markdown(f"**Flagged {is_anom.sum()} anomalies** from {len(signal)} readings using Z > {z_thresh}σ")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — CLUSTERING
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[2]:
        section("📊", "Clustering — Finding Natural Groups Without Labels")
        st.markdown("""
**Clustering** is **unsupervised** — it groups data points by similarity
without being told what the groups should be.

**Engineering uses:**
- **Customer segmentation** → target retention offers to each group differently
- **Device/machine grouping** → similar operating profiles share maintenance schedules
- **Fault mode discovery** → cluster failure events to find distinct failure types
- **Log analysis** → group similar error messages automatically
        """)

        algo = st.selectbox("Algorithm:", [
            "K-Means — Centroid-based (most common)",
            "DBSCAN — Density-based (handles arbitrary shapes)",
            "Hierarchical — Tree-based (shows merging structure)",
        ])

        np.random.seed(42)
        col1, col2 = st.columns([2, 1])

        with col2:
            st.markdown("#### Controls")
            dataset_shape = st.selectbox("Data shape:", [
                "Blobs (K-Means ideal)",
                "Rings (K-Means struggles)",
                "Crescents (DBSCAN shines)",
            ])
            if "K-Means" in algo:
                k = st.slider("K (clusters):", 2, 8, 4)
            elif "DBSCAN" in algo:
                eps = st.slider("ε (neighbourhood radius):", 0.1, 3.0, 0.8, step=0.1)
                min_samples = st.slider("Min samples:", 2, 15, 5)
            else:
                linkage = st.selectbox("Linkage:", ["ward", "complete", "average", "single"])
                k_hier = st.slider("Cut into K clusters:", 2, 8, 4)

        with col1:
            if "Blobs" in dataset_shape:
                X = np.vstack([
                    np.random.normal([2, 2], 0.5, (80, 2)),
                    np.random.normal([7, 3], 0.7, (100, 2)),
                    np.random.normal([4, 7], 0.6, (90, 2)),
                    np.random.normal([9, 8], 0.5, (80, 2)),
                ])
            elif "Rings" in dataset_shape:
                from sklearn.datasets import make_circles
                X, _ = make_circles(n_samples=350, factor=0.4, noise=0.05)
                X *= 5
            else:
                from sklearn.datasets import make_moons
                X, _ = make_moons(n_samples=350, noise=0.08)
                X *= 4

            if "K-Means" in algo:
                from sklearn.cluster import KMeans
                model = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = model.fit_predict(X)
                centers = model.cluster_centers_

                df_c = pd.DataFrame({"X": X[:,0], "Y": X[:,1], "Cluster": labels.astype(str)})
                fig = px.scatter(df_c, x="X", y="Y", color="Cluster",
                                 opacity=0.7, color_discrete_sequence=px.colors.qualitative.Set2)
                fig.add_trace(go.Scatter(x=centers[:,0], y=centers[:,1],
                                         mode="markers", marker=dict(size=18, symbol="x", color="white", line_width=2),
                                         name="Centroids"))
                fig = fig_defaults(fig, f"K-Means (K={k})", height=420)
                st.plotly_chart(fig, use_container_width=True)

                c1, c2, c3 = st.columns(3)
                c1.metric("Inertia (WCSS)", f"{model.inertia_:.1f}")
                c2.metric("Clusters", k)
                c3.metric("Points", len(X))

                st.markdown("#### Elbow Method — How to choose K")
                inertias = []
                for ki in range(1, 9):
                    km_i = KMeans(n_clusters=ki, random_state=42, n_init=10).fit(X)
                    inertias.append(km_i.inertia_)
                fig_elbow = go.Figure(go.Scatter(x=list(range(1, 9)), y=inertias, mode="lines+markers",
                                                  line=dict(color="#3b82f6", width=2),
                                                  marker=dict(size=8, color="#3b82f6")))
                fig_elbow.add_vline(x=k, line_dash="dash", line_color="#f59e0b",
                                    annotation_text=f"Your K={k}")
                fig_elbow = fig_defaults(fig_elbow, "Elbow Curve — choose K at the 'elbow'", height=280)
                st.plotly_chart(fig_elbow, use_container_width=True)

            elif "DBSCAN" in algo:
                from sklearn.cluster import DBSCAN
                model = DBSCAN(eps=eps, min_samples=min_samples)
                labels = model.fit_predict(X)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = (labels == -1).sum()

                df_c = pd.DataFrame({"X": X[:,0], "Y": X[:,1],
                                     "Cluster": [f"Cluster {l}" if l >= 0 else "Noise" for l in labels]})
                fig = px.scatter(df_c, x="X", y="Y", color="Cluster", opacity=0.8,
                                 color_discrete_sequence=px.colors.qualitative.Set2 + ["#6b7280"])
                fig = fig_defaults(fig, f"DBSCAN (ε={eps}, minPts={min_samples})", height=420)
                st.plotly_chart(fig, use_container_width=True)

                c1, c2, c3 = st.columns(3)
                c1.metric("Clusters found", n_clusters)
                c2.metric("Noise points", n_noise)
                c3.metric("Noise %", f"{n_noise/len(X)*100:.1f}%")

                info("""
<strong>DBSCAN advantage:</strong> finds clusters of any shape and marks outliers as noise.
No need to pre-specify K. Perfect for discovering fault mode clusters in sensor data
where cluster shapes are unknown and true anomalies exist.
                """)

            else:
                from sklearn.cluster import AgglomerativeClustering
                from scipy.cluster.hierarchy import dendrogram, linkage as sp_linkage
                model = AgglomerativeClustering(n_clusters=k_hier, linkage=linkage)
                labels = model.fit_predict(X)

                df_c = pd.DataFrame({"X": X[:,0], "Y": X[:,1], "Cluster": labels.astype(str)})
                fig = px.scatter(df_c, x="X", y="Y", color="Cluster", opacity=0.7,
                                 color_discrete_sequence=px.colors.qualitative.Set2)
                fig = fig_defaults(fig, f"Hierarchical Clustering (linkage={linkage}, K={k_hier})", height=420)
                st.plotly_chart(fig, use_container_width=True)

                info("""
<strong>Hierarchical clustering</strong> builds a dendrogram (tree) by progressively merging the
closest clusters. You can cut the tree at any level to get any K — no need to commit to K upfront.
Ideal for exploring how fault types relate to each other.
                """)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 — TIME SERIES FORECASTING
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[3]:
        section("🔮", "Time Series Forecasting — Predicting the Future from the Past")
        st.markdown("""
A time series is a sequence of values indexed by time.
Forecasting predicts future values from historical patterns by decomposing the signal into:

| Component | Description | Example |
|---|---|---|
| **Trend** | Long-term direction | Gradually rising energy demand |
| **Seasonality** | Repeating cycles | Daily temperature peak at noon |
| **Residual/Noise** | Random fluctuation | Sensor measurement noise |
| **Cyclical** | Irregular longer cycles | Business cycles, machine wear |

**Industry examples:** demand forecasting, energy load prediction,
predictive maintenance (predict time-to-failure), network capacity planning.
        """)

        col1, col2 = st.columns([2, 1])
        with col2:
            st.markdown("#### Signal Builder")
            trend_slope   = st.slider("Trend slope:", -0.5, 0.5, 0.15, step=0.05)
            season_amp    = st.slider("Seasonal amplitude:", 0.0, 30.0, 12.0, step=1.0)
            season_period = st.slider("Season period (days):", 7, 60, 30)
            noise_level   = st.slider("Noise level:", 0.0, 10.0, 3.0, step=0.5)
            horizon       = st.slider("Forecast horizon (days):", 7, 90, 30)
            method        = st.selectbox("Model:", [
                "Linear Regression (trend only)",
                "Linear + Seasonality",
                "Exponential Smoothing",
                "Moving Average",
            ])

        with col1:
            np.random.seed(99)
            n = 180
            t_idx = np.arange(n)
            t_dates = pd.date_range("2026-01-01", periods=n, freq="D")
            trend_vals   = 100 + trend_slope * t_idx
            season_vals  = season_amp * np.sin(2 * np.pi * t_idx / season_period)
            noise_vals   = np.random.normal(0, noise_level, n)
            y = trend_vals + season_vals + noise_vals

            t_fut_idx   = np.arange(n, n + horizon)
            t_fut_dates = pd.date_range(t_dates[-1] + pd.Timedelta("1D"), periods=horizon, freq="D")

            from sklearn.linear_model import LinearRegression

            if "Seasonality" in method:
                sin_feat = np.sin(2 * np.pi * t_idx / season_period)
                cos_feat = np.cos(2 * np.pi * t_idx / season_period)
                X_tr = np.column_stack([t_idx, sin_feat, cos_feat])
                sin_f = np.sin(2 * np.pi * t_fut_idx / season_period)
                cos_f = np.cos(2 * np.pi * t_fut_idx / season_period)
                X_fut = np.column_stack([t_fut_idx, sin_f, cos_f])
                lr = LinearRegression().fit(X_tr, y)
                y_pred = lr.predict(X_fut)

            elif "Linear" in method:
                lr = LinearRegression().fit(t_idx.reshape(-1, 1), y)
                y_pred = lr.predict(t_fut_idx.reshape(-1, 1))

            elif "Exponential" in method:
                alpha = 0.3
                smoothed = [y[0]]
                for val in y[1:]:
                    smoothed.append(alpha * val + (1 - alpha) * smoothed[-1])
                last = smoothed[-1]
                # Simple level forecast
                growth = (smoothed[-1] - smoothed[max(0, len(smoothed)-7)]) / 7
                y_pred = np.array([last + growth * i for i in range(1, horizon + 1)])

            else:  # Moving average
                window = min(14, n // 4)
                ma = np.convolve(y, np.ones(window) / window, mode="valid")
                last_ma = ma[-1]
                y_pred = np.full(horizon, last_ma)

            residuals = y - LinearRegression().fit(t_idx.reshape(-1,1), y).predict(t_idx.reshape(-1,1))
            ci = np.std(residuals) * 1.96 * np.sqrt(np.arange(1, horizon + 1) / n)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t_dates, y=y, name="Historical",
                                     line=dict(color="#3b82f6", width=1.5)))
            fig.add_trace(go.Scatter(x=t_fut_dates, y=y_pred, name="Forecast",
                                     line=dict(color="#f59e0b", width=2.5, dash="dash")))
            fig.add_trace(go.Scatter(
                x=list(t_fut_dates) + list(t_fut_dates[::-1]),
                y=list(y_pred + ci) + list((y_pred - ci)[::-1]),
                fill="toself", fillcolor="rgba(245,158,11,0.12)",
                line=dict(width=0), name="95% CI"
            ))
            fig = fig_defaults(fig, f"Time Series Forecast — {method}", height=380)
            st.plotly_chart(fig, use_container_width=True)

        # Decomposition visualisation
        st.markdown("### Signal Decomposition — What your model actually learns")
        fig_dec = go.Figure()
        fig_dec = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                subplot_titles=["Trend", "Seasonality", "Noise (Residual)"])
        fig_dec.add_trace(go.Scatter(x=t_dates, y=trend_vals, line=dict(color="#10b981")), row=1, col=1)
        fig_dec.add_trace(go.Scatter(x=t_dates, y=season_vals, line=dict(color="#f59e0b")), row=2, col=1)
        fig_dec.add_trace(go.Scatter(x=t_dates, y=noise_vals, line=dict(color="#ef4444", width=0.8)), row=3, col=1)
        fig_dec.update_layout(template="plotly_dark", height=400, paper_bgcolor="#1e293b",
                               plot_bgcolor="#0f172a", showlegend=False,
                               font=dict(color="#94a3b8"), margin=dict(l=40, r=20, t=50, b=20))
        st.plotly_chart(fig_dec, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 5 — DIMENSIONALITY REDUCTION
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[4]:
        section("🧮", "Dimensionality Reduction — Compressing Without Losing Signal")
        st.markdown("""
High-dimensional data is hard to visualise, slow to train on, and full of noise.
Dimensionality reduction finds a **lower-dimensional representation** that preserves
the most important structure.

**When you need it:**
- 500 sensor features → 2D scatter plot for exploration
- 10,000 word embeddings → compressed representation for a classifier
- Remove correlated/redundant features before ML training
- Visualise whether clusters exist before running K-Means
        """)

        algo_dr = st.selectbox("Algorithm:", [
            "PCA — Linear projection (fast, interpretable)",
            "t-SNE — Non-linear (great for visualisation)",
            "UMAP — Non-linear (fast + preserves global structure)",
        ])

        col1, col2 = st.columns([2, 1])
        with col2:
            st.markdown("#### Controls")
            n_dims = st.slider("Original dimensions:", 3, 20, 8)
            n_classes = st.slider("Classes in data:", 2, 5, 3)
            separation = st.slider("Class separation:", 0.5, 5.0, 2.5, step=0.5)

        with col1:
            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA

            np.random.seed(42)
            n_per = 100
            X_parts = []
            y_parts = []
            for c in range(n_classes):
                center = np.random.randn(n_dims) * separation
                X_parts.append(np.random.randn(n_per, n_dims) + center)
                y_parts.extend([c] * n_per)
            X_hd = np.vstack(X_parts)
            y_hd = np.array(y_parts)

            scaler = StandardScaler()
            X_sc = scaler.fit_transform(X_hd)

            if "PCA" in algo_dr:
                pca = PCA(n_components=2)
                X_2d = pca.fit_transform(X_sc)
                variance = pca.explained_variance_ratio_

                fig = plot_pca_2d(X_2d, y_hd)
                st.plotly_chart(fig, use_container_width=True)

                c1, c2, c3 = st.columns(3)
                c1.metric("PC1 variance", f"{variance[0]*100:.1f}%")
                c2.metric("PC2 variance", f"{variance[1]*100:.1f}%")
                c3.metric("Total captured", f"{sum(variance)*100:.1f}%")

                # Scree plot
                pca_full = PCA(n_components=min(n_dims, 10))
                pca_full.fit(X_sc)
                cumvar = np.cumsum(pca_full.explained_variance_ratio_) * 100
                fig_sc = go.Figure()
                fig_sc.add_trace(go.Bar(x=[f"PC{i+1}" for i in range(len(cumvar))],
                                        y=pca_full.explained_variance_ratio_ * 100,
                                        name="Per-component", marker_color="#3b82f6"))
                fig_sc.add_trace(go.Scatter(x=[f"PC{i+1}" for i in range(len(cumvar))],
                                            y=cumvar, name="Cumulative",
                                            line=dict(color="#f59e0b", width=2), yaxis="y2"))
                fig_sc.update_layout(yaxis2=dict(overlaying="y", side="right", title="Cumulative %"),
                                     template="plotly_dark", height=280,
                                     paper_bgcolor="#1e293b", plot_bgcolor="#0f172a",
                                     font=dict(color="#94a3b8"), margin=dict(l=40, r=60, t=40, b=30))
                st.plotly_chart(fig_sc, use_container_width=True)
                info("The <strong>scree plot</strong> helps choose how many PCs to keep. The 'elbow' is where variance gain starts to plateau.")

            elif "t-SNE" in algo_dr:
                from sklearn.manifold import TSNE
                perplexity = st.slider("Perplexity (neighbourhood size):", 5, 50, 30)
                tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=500)
                X_2d = tsne.fit_transform(X_sc)
                df_t = pd.DataFrame({"Dim1": X_2d[:,0], "Dim2": X_2d[:,1], "Class": y_hd.astype(str)})
                fig = px.scatter(df_t, x="Dim1", y="Dim2", color="Class",
                                 color_discrete_sequence=px.colors.qualitative.Set2, opacity=0.8)
                fig = fig_defaults(fig, f"t-SNE (perplexity={perplexity})", height=420)
                st.plotly_chart(fig, use_container_width=True)
                info("""
<strong>t-SNE</strong> preserves local neighbourhoods — nearby points in high-D stay nearby in 2D.
<strong>Distances between clusters are NOT meaningful.</strong> Use it for visualisation only, not as input to ML.
Perplexity ≈ expected neighbourhood size. Try 5–50.
                """)

            else:
                try:
                    from umap import UMAP
                    n_neighbors = st.slider("n_neighbors:", 5, 50, 15)
                    min_dist = st.slider("min_dist:", 0.0, 0.99, 0.1, step=0.05)
                    reducer = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
                    X_2d = reducer.fit_transform(X_sc)
                    df_u = pd.DataFrame({"Dim1": X_2d[:,0], "Dim2": X_2d[:,1], "Class": y_hd.astype(str)})
                    fig = px.scatter(df_u, x="Dim1", y="Dim2", color="Class",
                                     color_discrete_sequence=px.colors.qualitative.Set2, opacity=0.8)
                    fig = fig_defaults(fig, "UMAP Projection", height=420)
                    st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    warn("UMAP not installed. Run <code>pip install umap-learn</code>. Falling back to PCA.")
                    pca = PCA(n_components=2)
                    X_2d = pca.fit_transform(X_sc)
                    fig = plot_pca_2d(X_2d, y_hd)
                    st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 6 — NLP PIPELINE
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[5]:
        section("📝", "NLP — Teaching Machines to Understand Language")
        st.markdown("""
NLP converts **unstructured text** into structured numerical representations
that ML models can learn from.

**Engineering applications:**
- **Maintenance log classification** → auto-route to the right team
- **Incident report analysis** → extract root cause entities
- **Customer feedback** → sentiment + topic extraction
- **Compliance documents** → information retrieval, Q&A
        """)

        nlp_demo = st.selectbox("Choose NLP technique:", [
            "📊 Bag of Words & TF-IDF (the foundation)",
            "🔢 Word Embeddings (words as vectors)",
            "🏷️ Text Classification (ML on text)",
            "💬 Sentiment Analysis (rule + ML comparison)",
            "🔍 Named Entity Recognition (info extraction)",
        ])

        if "Bag of Words" in nlp_demo:
            st.markdown("### Bag of Words → TF-IDF")
            st.markdown("""
**Bag of Words (BoW):** Represent text as a vector of word counts.
Order is ignored — just *which* words appear and *how often*.

**Problem with BoW:** Common words like "the", "is", "at" dominate even though they carry no signal.

**TF-IDF fix:** Downweight words that appear in every document (low information),
upweight words that are rare and specific.

```
TF (term frequency)  = count of word in document / total words in document
IDF (inverse doc freq) = log(total documents / documents containing word)
TF-IDF score = TF × IDF
```
            """)

            docs = [
                "Bearing failure detected in engine ENG-L-001 compressor section",
                "Vibration levels nominal on all engines regular inspection passed",
                "Critical bearing wear detected immediate maintenance bearing replacement required",
                "Engine performance nominal all parameters within limits no issues found",
                "Fuel nozzle clogging detected engine ENG-R-001 fuel flow reduced",
            ]
            st.markdown("**Sample maintenance logs:**")
            for i, d in enumerate(docs):
                st.markdown(f"- `Doc {i+1}:` {d}")

            from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Bag of Words (raw counts)")
                cv = CountVectorizer(stop_words="english")
                bow = cv.fit_transform(docs)
                bow_df = pd.DataFrame(bow.toarray(), columns=cv.get_feature_names_out(),
                                      index=[f"Doc {i+1}" for i in range(len(docs))])
                st.dataframe(bow_df, use_container_width=True)

            with col2:
                st.markdown("#### TF-IDF (weighted)")
                tfidf = TfidfVectorizer(stop_words="english")
                tfidf_mat = tfidf.fit_transform(docs)
                tfidf_df = pd.DataFrame(tfidf_mat.toarray().round(3),
                                         columns=tfidf.get_feature_names_out(),
                                         index=[f"Doc {i+1}" for i in range(len(docs))])
                st.dataframe(tfidf_df, use_container_width=True)

            st.markdown("#### Top TF-IDF words per document")
            for i, row in tfidf_df.iterrows():
                top = row.sort_values(ascending=False).head(4)
                words = ", ".join([f"**{w}** ({v:.2f})" for w, v in top.items()])
                st.markdown(f"- `{i}`: {words}")

            info("Notice: 'bearing' scores high in Doc 1 and Doc 3 (rare across all docs) but 'detected' scores lower (appears in multiple docs). This is TF-IDF working correctly.")

        elif "Word Embeddings" in nlp_demo:
            st.markdown("### Word Embeddings — Words as Points in Vector Space")
            st.markdown("""
Instead of one-hot / count vectors, word embeddings map words to **dense vectors**
where **similar words cluster together in space**.

```
King  → [0.82, -0.23, 0.14, 0.76, ...]  (300 dimensions)
Queen → [0.80, -0.19, 0.18, 0.74, ...]  (nearby!)
Bolt  → [-0.32, 0.71, -0.45, 0.12, ...]  (far away)

Famous analogy:
King - Man + Woman ≈ Queen
Paris - France + Italy ≈ Rome
```

**Why this matters:** A model trained on Word2Vec embeddings "knows" that
"bearing failure" and "bearing wear" are semantically similar —
even if neither phrase appeared in your training data.
            """)

            # Simulate 2D word embedding space
            np.random.seed(42)
            maintenance_words = {
                "failure":     [0.8, 0.7],   "fault":   [0.75, 0.65],
                "breakdown":   [0.85, 0.6],  "crack":   [0.6, 0.8],
                "wear":        [0.7, 0.75],  "damage":  [0.65, 0.72],
                "nominal":     [-0.7, -0.6], "normal":  [-0.75, -0.65],
                "optimal":     [-0.8, -0.7], "stable":  [-0.65, -0.75],
                "bearing":     [0.3, 0.5],   "engine":  [0.1, 0.2],
                "vibration":   [0.4, 0.3],   "sensor":  [0.0, 0.1],
                "temperature": [0.2, 0.35],  "pressure": [0.15, 0.4],
            }
            categories = {
                "failure": "Failure terms", "fault": "Failure terms", "breakdown": "Failure terms",
                "crack": "Failure terms", "wear": "Failure terms", "damage": "Failure terms",
                "nominal": "Healthy terms", "normal": "Healthy terms", "optimal": "Healthy terms",
                "stable": "Healthy terms",
                "bearing": "Components", "engine": "Components", "vibration": "Components",
                "sensor": "Components", "temperature": "Components", "pressure": "Components",
            }
            # Add noise
            words_list = list(maintenance_words.keys())
            pts = np.array([maintenance_words[w] for w in words_list])
            pts += np.random.normal(0, 0.08, pts.shape)
            cats = [categories[w] for w in words_list]

            df_emb = pd.DataFrame({"x": pts[:,0], "y": pts[:,1], "word": words_list, "category": cats})
            fig = px.scatter(df_emb, x="x", y="y", text="word", color="category",
                             color_discrete_sequence=["#ef4444", "#10b981", "#3b82f6"],
                             size=[12]*len(words_list))
            fig.update_traces(textposition="top center", textfont_size=10)
            fig = fig_defaults(fig, "Word Embedding Space (2D visualisation)", height=480)
            st.plotly_chart(fig, use_container_width=True)
            success("Notice: failure-related words cluster together (top-right), healthy terms cluster (bottom-left), and components form their own group.")

        elif "Text Classification" in nlp_demo:
            st.markdown("### Text Classification — Routing Maintenance Tickets Automatically")
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import classification_report

            train_data = [
                ("Bearing making grinding noise", "Mechanical"),
                ("EGT temperature spike detected", "Thermal"),
                ("Vibration above threshold", "Mechanical"),
                ("Fuel flow reduced", "Fuel System"),
                ("Exhaust temperature warning", "Thermal"),
                ("Shaft bearing wear detected", "Mechanical"),
                ("Oil temperature high", "Thermal"),
                ("Fuel nozzle clogged", "Fuel System"),
                ("Hydraulic pressure low", "Hydraulic"),
                ("Landing gear actuator slow", "Hydraulic"),
                ("Compressor stall detected", "Fuel System"),
                ("Hot section inspection due", "Thermal"),
                ("Gear box vibration alert", "Mechanical"),
                ("Hydraulic fluid leak", "Hydraulic"),
                ("Combustion chamber crack", "Mechanical"),
            ]
            train_texts, train_labels = zip(*train_data)

            tfidf = TfidfVectorizer(ngram_range=(1, 2))
            X_tr = tfidf.fit_transform(train_texts)
            clf = LogisticRegression(max_iter=500, random_state=42)
            clf.fit(X_tr, train_labels)

            test_input = st.text_input(
                "Enter a maintenance ticket to classify:",
                "Engine shaft bearing shows unusual vibration and heat"
            )
            if test_input.strip():
                X_test = tfidf.transform([test_input])
                pred = clf.predict(X_test)[0]
                probs = clf.predict_proba(X_test)[0]
                classes = clf.classes_

                success(f"<strong>Predicted category: {pred}</strong>")

                prob_df = pd.DataFrame({"Category": classes, "Probability": probs.round(3)}).sort_values("Probability", ascending=False)
                fig = px.bar(prob_df, x="Category", y="Probability", color="Category",
                             color_discrete_sequence=px.colors.qualitative.Set2)
                fig = fig_defaults(fig, "Classification Probabilities", height=280)
                st.plotly_chart(fig, use_container_width=True)

            info("This uses <strong>TF-IDF + Logistic Regression</strong> — a simple but powerful baseline. In production, replace with a fine-tuned BERT model for much higher accuracy on domain-specific text.")

        elif "Sentiment" in nlp_demo:
            st.markdown("### Sentiment Analysis — Rule-based vs ML comparison")
            import re

            maintenance_texts = [
                "Outstanding engine performance, all readings nominal after overhaul",
                "Critical bearing failure detected, immediate shutdown required",
                "Average inspection results, nothing unusual to report",
                "Excellent vibration profile post maintenance, very impressed",
                "Severe fuel leak detected in left engine nacelle, dangerous",
                "System operating within parameters, no anomalies found",
                "Concerning temperature trends observed in turbine section",
                "Perfect compliance test, system fully operational and healthy",
            ]
            user_text = st.text_area("Add your own log entry:", "The engine shows some vibration but overall performance is acceptable")
            all_texts = maintenance_texts + ([user_text] if user_text.strip() else [])

            # Rule-based
            POS = {"outstanding", "excellent", "perfect", "nominal", "operational", "healthy", "impressed", "good", "well", "acceptable"}
            NEG = {"critical", "failure", "severe", "dangerous", "concerning", "fault", "problem", "leak", "shutdown", "anomaly"}

            def rule_sentiment(text):
                words = set(re.findall(r'\w+', text.lower()))
                p = len(words & POS); n = len(words & NEG)
                sc = (p - n) / max(p + n, 1)
                return sc, "Positive" if sc > 0.1 else "Negative" if sc < -0.1 else "Neutral"

            # Naive Bayes ML
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.naive_bayes import MultinomialNB
            train_texts_s = [
                "failure detected critical shutdown dangerous",
                "excellent performance nominal healthy optimal",
                "anomaly warning concerning problem issue",
                "outstanding results perfect system operational",
                "severe damage fault emergency stop now",
                "good inspection passed all parameters normal",
            ]
            train_labels_s = ["Negative","Positive","Negative","Positive","Negative","Positive"]
            tv = TfidfVectorizer()
            nb = MultinomialNB()
            nb.fit(tv.fit_transform(train_texts_s), train_labels_s)

            results = []
            for text in all_texts:
                rs, rl = rule_sentiment(text)
                ml_prob = nb.predict_proba(tv.transform([text]))[0]
                ml_pred = nb.classes_[ml_prob.argmax()]
                results.append({
                    "Text": text[:55] + "...",
                    "Rule Score": round(rs, 2),
                    "Rule Label": rl,
                    "ML Label": ml_pred,
                    "ML Confidence": f"{ml_prob.max()*100:.0f}%",
                    "Agreement": "✅" if rl == ml_pred or (rl == "Neutral") else "❌",
                })

            df_s = pd.DataFrame(results)
            st.dataframe(df_s, use_container_width=True, hide_index=True)

            col1, col2 = st.columns(2)
            with col1:
                fig_r = px.histogram(df_s, x="Rule Label", color="Rule Label",
                                     color_discrete_map={"Positive":"#10b981","Negative":"#ef4444","Neutral":"#94a3b8"},
                                     title="Rule-based Distribution")
                fig_r = fig_defaults(fig_r, "Rule-based Sentiment", height=250)
                st.plotly_chart(fig_r, use_container_width=True)
            with col2:
                fig_m = px.histogram(df_s, x="ML Label", color="ML Label",
                                     color_discrete_map={"Positive":"#10b981","Negative":"#ef4444"},
                                     title="ML Distribution")
                fig_m = fig_defaults(fig_m, "ML Naive Bayes Sentiment", height=250)
                st.plotly_chart(fig_m, use_container_width=True)

        else:  # NER
            st.markdown("### Named Entity Recognition — Extracting Information from Text")
            st.markdown("""
NER identifies and classifies **entities** (names, locations, dates, part numbers, etc.)
in unstructured text. Used to auto-populate structured databases from free-text maintenance reports.
            """)
            import re

            PATTERNS = {
                "ENGINE_ID": r'\bENG-[A-Z]-\d{3}\b',
                "PART_NUMBER": r'\b[A-Z]{2,4}-\d{4,6}\b',
                "TEMPERATURE": r'\b\d{2,4}\s*°?[CF]\b',
                "DATE": r'\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{4}\b',
                "MEASUREMENT": r'\b\d+\.?\d*\s*(psi|rpm|kPa|g|mm|inches|hrs?|hours?|°C|°F)\b',
                "SEVERITY": r'\b(critical|severe|warning|nominal|normal|urgent)\b',
            }
            COLOR_MAP = {
                "ENGINE_ID": "#3b82f6", "PART_NUMBER": "#8b5cf6",
                "TEMPERATURE": "#ef4444", "DATE": "#10b981",
                "MEASUREMENT": "#f59e0b", "SEVERITY": "#ec4899",
            }

            sample_report = """Inspection report 2026-06-21: Engine ENG-L-001 shows bearing wear
on component BRG-00423. EGT reading 698°C exceeds normal 640°C limit by 58°C.
Vibration level 1.87g measured at fan shaft. Part BR-007892 requires replacement
within 200 hrs. Severity: critical. Next inspection due 2026-07-15.
Engineer authorised: shutdown ENG-R-001 immediately for inspection."""

            user_report = st.text_area("Maintenance report:", sample_report, height=120)

            entities_found = []
            annotated = user_report
            for entity_type, pattern in PATTERNS.items():
                for match in re.finditer(pattern, user_report, re.IGNORECASE):
                    entities_found.append({
                        "Entity": match.group(),
                        "Type": entity_type,
                        "Start": match.start(),
                        "End": match.end(),
                    })

            entities_found.sort(key=lambda x: x["Start"])

            if entities_found:
                st.markdown("#### Extracted Entities")
                df_ent = pd.DataFrame(entities_found).drop(columns=["Start","End"])
                df_ent["Color"] = df_ent["Type"].map(COLOR_MAP)
                st.dataframe(df_ent[["Type","Entity"]], use_container_width=True, hide_index=True)

                for etype in PATTERNS:
                    matches = [e["Entity"] for e in entities_found if e["Type"] == etype]
                    if matches:
                        color = COLOR_MAP[etype]
                        badges = " ".join([f'<span style="background:{color}22;border:1px solid {color};border-radius:4px;padding:2px 6px;font-size:0.8rem;color:{color};margin:2px">{m}</span>' for m in matches])
                        st.markdown(f"**{etype}:** {badges}", unsafe_allow_html=True)

                st.markdown("#### Structured Output (auto-populated database record)")
                structured = {t: [e["Entity"] for e in entities_found if e["Type"] == t] for t in PATTERNS}
                structured_clean = {k: v[0] if len(v) == 1 else v for k, v in structured.items() if v}
                st.json(structured_clean)

                info("In production, this structured JSON would auto-populate your CMMS (Computerised Maintenance Management System), create work orders, and update asset history — no manual data entry.")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 7 — NEURAL NETWORKS
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[6]:
        section("🧬", "Neural Networks — How Deep Learning Actually Works")
        st.markdown("""
A neural network is a **function approximator** built from layers of
simple mathematical operations (linear transformation + non-linear activation).
It learns by adjusting millions of numbers (weights) to minimise prediction error.
        """)

        nn_topic = st.selectbox("Deep dive into:", [
            "🏗️ Network Architecture — layers, neurons, activations",
            "📉 Training — forward pass, loss, backpropagation",
            "🖼️ CNN — how images become predictions",
            "⏱️ LSTM — how sequences are remembered",
            "🔄 Transformer — the architecture behind LLMs",
        ])

        if "Architecture" in nn_topic:
            st.code("""
NEURAL NETWORK ANATOMY
──────────────────────────────────────────────────────────────
INPUT LAYER        HIDDEN LAYERS           OUTPUT LAYER
(raw features)     (learned representations)  (predictions)

[temp=640]   ──►  Neuron computes:        ──►  [P(failure)=0.73]
[vibr=0.42]  ──►  z = Σ(weight × input)  ──►  [P(normal)=0.27]
[rpm=3100]   ──►    + bias
[pressure]   ──►  a = activation(z)
             ──►
                   Popular activations:
                   ReLU(z) = max(0, z)      ← most common
                   Sigmoid(z) = 1/(1+e⁻ᶻ)  ← for probabilities
                   Tanh(z) = (eᶻ-e⁻ᶻ)/(eᶻ+e⁻ᶻ) ← centered at 0
                   Softmax(z) = eᶻ/Σeᶻ     ← multi-class output

DEPTH = number of hidden layers
WIDTH = number of neurons per layer

Shallow (1-2 layers): simple patterns, fast to train
Deep (10+ layers):    complex patterns, needs more data
            """, language="text")

            # Live neural network demo
            st.markdown("### Live: Train a 2-layer network on industrial data")
            from sklearn.neural_network import MLPClassifier
            from sklearn.datasets import make_classification
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score

            col1, col2 = st.columns(2)
            with col1:
                hidden_size = st.slider("Neurons per layer:", 4, 64, 16)
                n_layers = st.slider("Hidden layers:", 1, 4, 2)
                activation_fn = st.selectbox("Activation:", ["relu", "tanh", "logistic"])
            with col2:
                learning_rate = st.select_slider("Learning rate:", [0.001, 0.01, 0.05, 0.1, 0.5], value=0.01)
                n_epochs = st.slider("Max iterations:", 50, 500, 200)

            X_nn, y_nn = make_classification(n_samples=500, n_features=6, n_informative=4,
                                             n_redundant=1, random_state=42)
            X_tr_nn, X_te_nn, y_tr_nn, y_te_nn = train_test_split(X_nn, y_nn, test_size=0.2, random_state=42)

            hidden_layers = tuple([hidden_size] * n_layers)
            mlp = MLPClassifier(hidden_layer_sizes=hidden_layers, activation=activation_fn,
                                learning_rate_init=learning_rate, max_iter=n_epochs,
                                random_state=42)
            mlp.fit(X_tr_nn, y_tr_nn)
            acc = accuracy_score(y_te_nn, mlp.predict(X_te_nn))

            c1, c2, c3 = st.columns(3)
            c1.metric("Test Accuracy", f"{acc*100:.1f}%")
            c2.metric("Layers", n_layers + 2)
            c3.metric("Total params (est.)", f"{6*hidden_size + (n_layers-1)*hidden_size**2 + hidden_size*2:,}")

            if hasattr(mlp, 'loss_curve_'):
                fig_loss = go.Figure(go.Scatter(y=mlp.loss_curve_, mode="lines",
                                                line=dict(color="#3b82f6", width=1.5)))
                fig_loss = fig_defaults(fig_loss, "Training Loss Curve", height=280)
                st.plotly_chart(fig_loss, use_container_width=True)

        elif "Training" in nn_topic:
            st.code("""
TRAINING LOOP — HOW A NEURAL NETWORK LEARNS
────────────────────────────────────────────────────────────────────────────

For each batch of training data:

  1. FORWARD PASS
     Input → Layer 1 → Layer 2 → ... → Output ŷ

  2. COMPUTE LOSS
     How wrong are we?
     ┌─────────────────────────────────────────────────────────┐
     │ Regression:       MSE = mean((y - ŷ)²)                │
     │ Binary class.:    BCE = -[y·log(ŷ) + (1-y)·log(1-ŷ)] │
     │ Multi-class:      CCE = -Σ y·log(ŷ)  (cross-entropy)  │
     └─────────────────────────────────────────────────────────┘

  3. BACKWARD PASS (backpropagation)
     Compute gradient: ∂Loss/∂weight for every weight
     Chain rule: propagate error signal backward through layers
     This tells each weight: "you need to change by this much"

  4. UPDATE WEIGHTS
     weight = weight - learning_rate × gradient
     ┌──────────────────────────────────────────────────────────┐
     │ SGD:    w -= lr × grad                   (simple)       │
     │ Adam:   w -= lr × m̂/(√v̂ + ε)   (adaptive, best)      │
     │ RMSProp: w -= lr × grad/√(v + ε) (between both)        │
     └──────────────────────────────────────────────────────────┘

  Repeat for N epochs until loss converges.

KEY HYPERPARAMETERS:
  learning_rate:  how big each step is (too big → diverge, too small → slow)
  batch_size:     how many samples per gradient update (32-256 typical)
  epochs:         how many full passes through the data
  optimizer:      Adam is almost always the best starting point
            """, language="text")

            st.markdown("### Visualise: Effect of Learning Rate")
            lrs = [0.001, 0.01, 0.05, 0.3]
            fig_lr = go.Figure()
            for lr_val in lrs:
                loss_curve = [10]
                for i in range(80):
                    noise = np.random.normal(0, 0.1) if lr_val < 0.1 else np.random.normal(0, 0.5)
                    next_loss = loss_curve[-1] * (1 - lr_val * 0.3) + noise
                    if lr_val >= 0.3 and i > 30:
                        next_loss = abs(next_loss) * (1 + np.random.uniform(0, 0.05))
                    loss_curve.append(max(0.1, next_loss))
                fig_lr.add_trace(go.Scatter(y=loss_curve, name=f"lr={lr_val}", mode="lines"))
            fig_lr = fig_defaults(fig_lr, "Loss curves for different learning rates", height=320)
            st.plotly_chart(fig_lr, use_container_width=True)
            info("lr=0.001 → slow but stable. lr=0.01 → good balance. lr=0.3 → diverges. <strong>Adam optimizer adapts the learning rate automatically</strong> — less tuning needed.")

        elif "CNN" in nn_topic:
            st.code("""
HOW A CNN PROCESSES AN IMAGE FOR DEFECT DETECTION
────────────────────────────────────────────────────────────────────────

Input image: 224 × 224 × 3  (height × width × RGB channels)
          ↓
CONV Layer 1: 32 filters, 3×3 kernel
  Each filter slides across the image, computing dot products
  Learns to detect: edges, corners, colour gradients
  Output: 222 × 222 × 32  (32 feature maps)
          ↓
MaxPool 2×2: keeps strongest activations, halves spatial size
  Output: 111 × 111 × 32
          ↓
CONV Layer 2: 64 filters, 3×3
  Combines edge detectors → detects textures, curves
  Output: 109 × 109 × 64
          ↓
MaxPool 2×2  → 54 × 54 × 64
          ↓
CONV Layer 3: 128 filters
  Combines textures → detects shapes, parts
  Output: 52 × 52 × 128
          ↓
MaxPool 2×2  → 26 × 26 × 128
          ↓
FLATTEN: 26 × 26 × 128 = 86,528 values
          ↓
FULLY CONNECTED: 512 neurons  (combines spatial features)
          ↓
DROPOUT: randomly zero 50% of neurons  (prevents overfitting)
          ↓
OUTPUT: 3 neurons + Softmax
  [P(normal)=0.03, P(scratch)=0.92, P(crack)=0.05]

Total parameters: ~2.5 million
Training time: ~2 hours on GPU (1000 images × 50 epochs)
            """, language="text")

            st.markdown("### Convolution Operation — Visualised")
            kernel_type = st.selectbox("Filter type:", ["Edge detector", "Blur", "Sharpen"])
            kernels = {
                "Edge detector": np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]),
                "Blur":          np.array([[1,1,1],[1,1,1],[1,1,1]]) / 9.0,
                "Sharpen":       np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]),
            }
            k = kernels[kernel_type]
            fig_k = px.imshow(k, text_auto=True, color_continuous_scale="RdBu",
                              title=f"{kernel_type} kernel (3×3)", aspect="equal")
            fig_k.update_layout(template="plotly_dark", height=280, paper_bgcolor="#1e293b",
                                 font=dict(color="#94a3b8"), margin=dict(t=40, b=20, l=20, r=20))
            st.plotly_chart(fig_k, use_container_width=True)
            info(f"This {kernel_type} kernel slides across the image, computing a weighted sum at each position. 64+ of these filters learned automatically during training — each detects a different visual pattern.")

        elif "LSTM" in nn_topic:
            st.code("""
HOW AN LSTM PROCESSES SENSOR TIME SERIES
────────────────────────────────────────────────────────────────────────────

Standard RNN problem: vanishing gradients → forgets long-term patterns

LSTM solution: 3 gating mechanisms that control memory

At each time step t (each sensor reading):
  Input: x_t  (current reading)
  State: h_t  (short-term memory), C_t (long-term memory cell)

  ┌─────────────────────────────────────────────────────────────────┐
  │ FORGET gate:  f_t = σ(Wf·[h_{t-1}, x_t] + bf)               │
  │               "How much of last memory to keep?"               │
  │                                                                 │
  │ INPUT gate:   i_t = σ(Wi·[h_{t-1}, x_t] + bi)               │
  │               C̃_t = tanh(Wc·[h_{t-1}, x_t] + bc)           │
  │               "What new info to add to memory?"                │
  │                                                                 │
  │ UPDATE cell:  C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t             │
  │               "New long-term memory"                            │
  │                                                                 │
  │ OUTPUT gate:  o_t = σ(Wo·[h_{t-1}, x_t] + bo)               │
  │               h_t = o_t ⊙ tanh(C_t)                          │
  │               "What to output as prediction"                   │
  └─────────────────────────────────────────────────────────────────┘

APPLICATION: Bearing failure prediction
  Input sequence: [vibr_t-100, vibr_t-99, ..., vibr_t]  (100 readings)
  Output: P(bearing_fails_in_next_24h)

  The LSTM "remembers" the gradual vibration increase over 100 steps
  and correlates it with past failures.
            """, language="text")

        else:  # Transformer
            st.code("""
THE TRANSFORMER — ARCHITECTURE BEHIND GPT-4, CLAUDE, BERT
────────────────────────────────────────────────────────────────────────────

Key innovation: SELF-ATTENTION
  Every token attends to every other token simultaneously
  (unlike LSTM which processes sequentially)

For a maintenance log: "bearing in engine ENG-L-001 shows wear"

  token:    bearing  in   engine  ENG-L-001  shows  wear

  Attention to "wear":
  bearing ──────────────────────────────────────────► 0.45 (highest)
  engine  ──────────────────────────────────────────► 0.28
  ENG     ──────────────────────────────────────────► 0.15
  in/the  ──────────────────────────────────────────► 0.02 (ignored)

  "wear" attends most to "bearing" → learns the relationship

ARCHITECTURE:
  Input → Tokenise → Embed → Positional Encoding
       ↓
  [Multi-Head Attention] × N layers
       ↓
  [Feed-Forward Network] × N layers
       ↓
  Output: next token probabilities (GPT)
       or: classification head (BERT)

SCALE:
  GPT-2:   117M parameters,  40GB text
  GPT-3:   175B parameters,  570GB text
  Claude:  Unknown, estimated 100B-1T parameters

"Attention is all you need" — Vaswani et al., 2017 (the paper that started it all)
            """, language="text")
            info("The Transformer's self-attention mechanism is why LLMs can understand long documents, write code, and reason across topics. Every major AI model (GPT, Claude, Gemini, Llama) uses this architecture.")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 8 — LLMs & RAG
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[7]:
        section("🤖", "LLMs & RAG — Grounding AI in Your Domain Knowledge")

        llm_topic = st.selectbox("Topic:", [
            "📚 How LLMs work — pre-training, fine-tuning, prompting",
            "🔍 RAG — Retrieval Augmented Generation (live demo)",
            "🎯 Prompt Engineering — getting the best from LLMs",
            "⚠️ LLM Limitations — hallucination, bias, data cutoff",
        ])

        if "How LLMs" in llm_topic:
            st.code("""
HOW A LARGE LANGUAGE MODEL IS BUILT
────────────────────────────────────────────────────────────────────────────

STAGE 1 — PRE-TRAINING  (expensive: weeks on 1000s of GPUs)
  Data:    Internet text, books, code (~1 trillion tokens)
  Task:    Predict next token given all previous tokens
  Result:  Model learns grammar, facts, reasoning, code, and more
  Output:  Base model (knows a lot, but doesn't follow instructions)

STAGE 2 — FINE-TUNING  (cheaper: hours/days)
  Option A: Supervised Fine-Tuning (SFT)
    Data:    Human-written (instruction, response) pairs
    Effect:  Model learns to follow instructions

  Option B: Domain Fine-Tuning
    Data:    Your company's manuals, reports, code
    Effect:  Model becomes expert in your domain

  Option C: RLHF (Reinforcement Learning from Human Feedback)
    Process: Humans rank outputs → reward model → RL training
    Effect:  Responses become safer, more helpful, less toxic
    Used by: Claude, ChatGPT

STAGE 3 — DEPLOYMENT  (inference)
  User sends prompt → model generates token by token (autoregressive)
  Temperature: 0.0 = deterministic, 1.0 = creative, 2.0 = chaotic
  Top-p/Top-k: sampling from the most likely next tokens

YOUR OPTIONS AS AN ENGINEER:
  1. API call       → Claude / GPT-4  (easiest, most capable)
  2. Fine-tune      → Domain-specific knowledge
  3. Local model    → Llama, Mistral  (privacy, no API cost)
  4. RAG            → Private docs without fine-tuning
            """, language="text")

        elif "RAG" in llm_topic:
            st.markdown("### RAG — Live Demo (no API key needed)")
            st.markdown("""
RAG = Retrieval Augmented Generation.
Instead of fine-tuning an LLM on your documents, you:
1. **Store** documents in a vector database
2. **Retrieve** the most relevant chunks for each query
3. **Inject** retrieved context into the LLM prompt
4. **Generate** an answer grounded in your documents
            """)

            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            KNOWLEDGE_BASE = [
                {"id": "KB-001", "source": "Maintenance Manual §4.2",
                 "text": "When EGT exceeds 680°C for more than 5 minutes, this indicates compressor fouling. Recommended action: schedule compressor wash within next 50 flight hours. Do not exceed 700°C."},
                {"id": "KB-002", "source": "Maintenance Manual §6.1",
                 "text": "Vibration levels above 1.0 g on the fan shaft indicate bearing wear. Levels above 2.0 g require immediate ground inspection. Replace bearing within 200 flight hours if trend is upward."},
                {"id": "KB-003", "source": "Fault Isolation Manual §12",
                 "text": "N1 fan speed variance greater than 5% indicates potential blade damage or FOD (Foreign Object Damage). Immediate borescope inspection is required. Do not continue flight."},
                {"id": "KB-004", "source": "Maintenance Manual §8.3",
                 "text": "Fuel flow 10% above baseline with normal thrust output indicates fuel nozzle clogging. Clean or replace nozzles at next scheduled maintenance. Monitor closely if difference exceeds 15%."},
                {"id": "KB-005", "source": "Operations Manual §3",
                 "text": "Oil temperature above 155°C indicates either oil cooler malfunction or excessive bearing wear. Check oil level and quality. If temperature exceeds 160°C, shut down engine."},
                {"id": "KB-006", "source": "Troubleshooting Guide §9",
                 "text": "EGT and N1 deviating together often indicates compressor stall. Reduce thrust immediately. If condition persists, shut down and perform full borescope inspection of all compressor stages."},
            ]

            vectorizer_rag = TfidfVectorizer(ngram_range=(1,2))
            doc_texts = [d["text"] for d in KNOWLEDGE_BASE]
            doc_vectors_rag = vectorizer_rag.fit_transform(doc_texts)

            def retrieve(query, top_k=2):
                q_vec = vectorizer_rag.transform([query])
                scores = cosine_similarity(q_vec, doc_vectors_rag)[0]
                top_idx = np.argsort(scores)[::-1][:top_k]
                return [(KNOWLEDGE_BASE[i], float(scores[i])) for i in top_idx]

            def simulate_llm_answer(query, context_docs):
                """Simulate an LLM answer using the retrieved context."""
                combined = " ".join([d["text"] for d, _ in context_docs])
                # Extract key numbers and recommendations
                import re
                numbers = re.findall(r'\d+\.?\d*\s*(?:°C|g|%|hrs?|hours?|minutes?)', combined)
                actions = [s.strip() for s in re.split(r'[.!]', combined)
                           if any(w in s.lower() for w in ["recommend","require","schedule","replace","inspect","shut","clean","monitor"])]
                answer = f"Based on the maintenance documentation:\n\n"
                if actions:
                    answer += "**Recommended actions:**\n"
                    for a in actions[:3]:
                        if a.strip():
                            answer += f"- {a.strip()}.\n"
                if numbers:
                    answer += f"\n**Key thresholds mentioned:** {', '.join(set(numbers[:6]))}"
                return answer

            st.markdown("#### Knowledge Base")
            for doc in KNOWLEDGE_BASE:
                with st.expander(f"📄 {doc['id']} — {doc['source']}"):
                    st.markdown(doc["text"])

            st.markdown("---")
            query = st.text_input(
                "Ask a maintenance question:",
                "Engine EGT is 695°C and vibration is 1.3g. What should I do?"
            )

            if query.strip():
                results = retrieve(query, top_k=2)

                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown("#### Step 1 — Retrieved Documents")
                    for doc, score in results:
                        st.markdown(f"""
                        <div style="background:#1e293b;border:1px solid {'#10b981' if score > 0.3 else '#334155'};
                                    border-radius:8px;padding:0.75rem;margin-bottom:0.5rem;">
                          <div style="font-size:0.75rem;color:#94a3b8;">{doc['source']} · Relevance: {score:.2f}</div>
                          <div style="font-size:0.85rem;color:#e2e8f0;margin-top:0.3rem;">{doc['text'][:160]}...</div>
                        </div>
                        """, unsafe_allow_html=True)

                with col2:
                    st.markdown("#### Step 2 — Generated Answer")
                    answer = simulate_llm_answer(query, results)
                    st.markdown(answer)
                    info("In production, Step 2 would send the retrieved context + query to Claude or GPT-4, getting a much richer answer. The RAG architecture ensures the answer is <strong>grounded in your manuals</strong>, not hallucinated.")

                st.markdown("#### Retrieval Score Breakdown")
                scores_all = cosine_similarity(vectorizer_rag.transform([query]), doc_vectors_rag)[0]
                score_df = pd.DataFrame({"Document": [d["id"] for d in KNOWLEDGE_BASE], "Relevance Score": scores_all.round(3)})
                fig_scores = px.bar(score_df, x="Document", y="Relevance Score",
                                    color="Relevance Score", color_continuous_scale="Viridis")
                fig_scores = fig_defaults(fig_scores, "Cosine Similarity to Query", height=280)
                st.plotly_chart(fig_scores, use_container_width=True)

        elif "Prompt Engineering" in llm_topic:
            st.markdown("### Prompt Engineering Patterns")
            patterns = {
                "Zero-shot": {
                    "desc": "Ask directly — no examples provided. Works for simple tasks where the model has strong priors.",
                    "example": 'Classify this maintenance log as Critical, Warning, or Normal:\n"Bearing vibration at 0.45g, temperature nominal"\n\nAnswer:',
                },
                "Few-shot": {
                    "desc": "Provide 2-5 examples before your question. Massively improves performance on domain-specific tasks.",
                    "example": 'Log: "EGT 698°C, vibration 1.8g" → Critical\nLog: "All parameters nominal" → Normal\nLog: "Fuel flow 8% above baseline" → Warning\nLog: "Bearing noise detected, vibration 0.9g" →',
                },
                "Chain-of-Thought": {
                    "desc": "Ask the model to 'think step by step'. Dramatically improves reasoning and multi-step problems.",
                    "example": 'Analyse this sensor data and explain your reasoning step by step:\nEGT: 695°C (limit: 680°C)\nVibration: 1.2g (limit: 1.0g)\nFuel flow: +12% above baseline\n\nStep 1: Check each parameter against limits...',
                },
                "Role Prompting": {
                    "desc": "Assign an expert role. Shifts the model's response style and knowledge application.",
                    "example": 'You are a senior aircraft maintenance engineer with 20 years of experience on Rolls-Royce Trent engines. A junior engineer asks:\n"What does it mean when EGT and vibration both exceed limits simultaneously?"',
                },
                "RAG Prompt": {
                    "desc": "Inject retrieved context. Grounds answers in your documents, preventing hallucination.",
                    "example": 'CONTEXT (from Maintenance Manual §4.2):\n"EGT above 680°C indicates compressor fouling. Max limit: 700°C."\n\nBased ONLY on the context above, answer:\nIs EGT of 695°C a concern? What action is required?',
                },
            }

            for pattern, details in patterns.items():
                with st.expander(f"**{pattern}**"):
                    st.markdown(f"**When to use:** {details['desc']}")
                    st.code(details["example"], language="text")

        else:
            st.markdown("### LLM Limitations — What Engineers Must Know")
            limitations = [
                ("🎭 Hallucination", "LLMs generate plausible-sounding but **factually wrong** answers with high confidence. An LLM might invent a maintenance procedure that doesn't exist.", "Always verify critical outputs. Use RAG to ground answers in real documents. Never trust LLM output for safety-critical decisions without human verification."),
                ("📅 Knowledge Cutoff", "LLMs are trained on data up to a specific date. They don't know about your latest product versions, recent incidents, or current regulations.", "Use RAG with your up-to-date documents. Fine-tune on recent domain data. Always ask the model about its knowledge cutoff when querying time-sensitive topics."),
                ("📏 Context Window", "LLMs can only 'read' a limited amount of text at once (4K–200K tokens). Longer documents get truncated or lose coherence.", "Chunk large documents. Use RAG to retrieve only relevant chunks. Summarise long histories. Monitor token counts in production."),
                ("🔢 Maths & Reasoning", "LLMs are not calculators. They can make arithmetic errors, especially with large numbers or multi-step calculations.", "Use tool calling to offload maths to Python. Validate numerical outputs. Chain-of-thought helps but doesn't guarantee correctness."),
                ("⚖️ Bias", "LLMs inherit biases from training data. They may perform worse on underrepresented domains, languages, or demographic groups.", "Evaluate on your specific domain. Test with diverse inputs. Fine-tune on representative data. Monitor for systematic errors."),
                ("🔒 Data Privacy", "Sending proprietary data to API-based LLMs means it leaves your infrastructure.", "Use local models (Llama, Mistral) for sensitive data. Review provider data policies. Consider private cloud deployment."),
            ]
            for title, problem, solution in limitations:
                with st.expander(title):
                    col1, col2 = st.columns(2)
                    col1.markdown(f"**Problem:** {problem}")
                    col2.markdown(f"**Mitigation:** {solution}")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 9 — COMPUTER VISION
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[8]:
        section("👁️", "Computer Vision — Teaching Machines to See")
        st.markdown("""
Computer vision enables machines to interpret and understand **visual information**
from images or video. In engineering, this replaces or augments human visual inspection.
        """)

        cv_topic = st.selectbox("Topic:", [
            "🖼️ Image Classification — is this defective?",
            "📦 Object Detection — where are the defects?",
            "📊 Pixel Statistics — what computer vision 'sees'",
            "🎭 Data Augmentation — making training data from less",
        ])

        if "Classification" in cv_topic:
            st.markdown("### CNN Defect Classification Pipeline")
            st.code("""
INDUSTRIAL DEFECT DETECTION PIPELINE
────────────────────────────────────────────────────────────────────────
1. IMAGE CAPTURE
   Camera: 12MP industrial camera, 100fps
   Lighting: Structured light / dark-field illumination
   Trigger: Part detected by proximity sensor

2. PREPROCESSING
   Resize: 224×224 pixels (ResNet input format)
   Normalise: pixel values 0→255 scaled to 0→1
   Augment:   Random flip, rotate ±15°, brightness ±20%
   (augmentation only during training, not inference)

3. MODEL: ResNet-50 (Transfer Learning)
   Pre-trained: ImageNet (1.2M images, 1000 classes)
   Fine-tuned:  Your defect dataset (500-5000 images)
   Output:      [P(OK)=0.03, P(scratch)=0.91, P(crack)=0.06]

4. DECISION
   If P(defect) > threshold → reject + flag for human review
   Threshold tuned to balance: false rejects vs missed defects

5. FEEDBACK LOOP
   Human reviewer labels rejected parts
   Correct labels → retrain model monthly
   New defect types → add to training set

PERFORMANCE (typical):
   Throughput: 60 parts/minute
   Accuracy:   97.3% (vs 89% human inspector)
   Consistency: 100% (no fatigue, no shift variation)
            """, language="text")

            st.markdown("### Confusion Matrix — Where the Model Makes Mistakes")
            np.random.seed(42)
            classes = ["OK", "Scratch", "Crack", "Void"]
            cm = np.array([
                [485, 8,  4,  3],
                [ 12, 91, 2,  0],
                [  5, 3, 87,  5],
                [  4, 0,  6, 90],
            ])
            fig_cm = go.Figure(go.Heatmap(
                z=cm, x=classes, y=classes,
                colorscale="Blues",
                text=cm, texttemplate="%{text}",
            ))
            fig_cm.update_layout(template="plotly_dark", height=400,
                                 paper_bgcolor="#1e293b", font=dict(color="#94a3b8"),
                                 margin=dict(l=40, r=20, t=40, b=40),
                                 xaxis_title="Predicted", yaxis_title="Actual")
            st.plotly_chart(fig_cm, use_container_width=True)

            total = cm.sum()
            correct = np.trace(cm)
            success(f"Overall accuracy: {correct/total*100:.1f}% | Most errors: OK classified as Scratch (8 parts) — adjust threshold to reduce false rejects.")

        elif "Detection" in cv_topic:
            st.markdown("### Object Detection — Locating Defects with Bounding Boxes")
            st.code("""
OBJECT DETECTION vs CLASSIFICATION
────────────────────────────────────────────────────────────────────────
Classification:  "This image contains a crack."         (one label)
Detection:       "There is a crack at [x=142, y=87,     (location)
                  width=45, height=12], confidence=0.93"

YOLO (You Only Look Once) — fastest detection architecture
  Single forward pass → detects all objects at once
  Speed: 30-100 FPS on GPU  (real-time inspection)

YOLOv8 output format:
  For each grid cell: [x, y, w, h, confidence, class_probs...]

  [0.45, 0.32, 0.08, 0.04, 0.93, 0.01, 0.95, 0.04]
   ^x    ^y    ^w    ^h    ^obj  ^OK   ^scr  ^crack
         (all relative to image size 0-1)

INDUSTRY EXAMPLE — PCB Inspection (electronics):
  Defects detected: solder bridges, missing components, lifted leads
  Speed: 1 board per second
  Resolution: 0.05mm per pixel
  Dataset needed: 500+ images per defect type
            """, language="text")

            # Simulate bounding box visualisation
            st.markdown("### Simulated Detection Output")
            np.random.seed(7)
            img_size = 400
            fig_det = go.Figure()
            fig_det.add_shape(type="rect", x0=0, y0=0, x1=img_size, y1=img_size,
                              fillcolor="#1e293b", line_color="#334155")

            defects = [
                (80, 100, 160, 140, "Scratch", 0.94, "#ef4444"),
                (220, 200, 290, 240, "Crack", 0.87, "#f59e0b"),
                (310, 60, 370, 110, "Void", 0.91, "#8b5cf6"),
            ]
            for x0, y0, x1, y1, label, conf, color in defects:
                fig_det.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                                  line=dict(color=color, width=2))
                fig_det.add_annotation(x=x0, y=y0 - 5, text=f"{label} {conf:.0%}",
                                       showarrow=False, font=dict(color=color, size=11),
                                       bgcolor="#0f172a", bordercolor=color)

            fig_det.add_annotation(x=img_size/2, y=img_size/2,
                                   text="[Component Surface Image]",
                                   showarrow=False, font=dict(color="#475569", size=14))
            fig_det.update_layout(template="plotly_dark", height=420,
                                  paper_bgcolor="#1e293b", plot_bgcolor="#0f172a",
                                  xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                  yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x"),
                                  margin=dict(l=10, r=10, t=30, b=10),
                                  title="YOLOv8 Detection Output (simulated)")
            st.plotly_chart(fig_det, use_container_width=True)

        elif "Pixel Statistics" in cv_topic:
            st.markdown("### What Computer Vision Actually 'Sees'")
            st.markdown("A 224×224 RGB image = 224 × 224 × 3 = **150,528 numbers**. Nothing else.")
            np.random.seed(42)
            img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
            img[10:20, 10:20] = [220, 40, 40]  # red defect region

            col1, col2, col3 = st.columns(3)
            for ch, (col, ch_name, color) in enumerate(zip([col1, col2, col3],
                                                             ["Red", "Green", "Blue"],
                                                             ["Reds", "Greens", "Blues"])):
                with col:
                    fig_ch = px.imshow(img[:,:,ch], color_continuous_scale=color,
                                       title=f"{ch_name} channel", zmin=0, zmax=255,
                                       aspect="equal")
                    fig_ch.update_layout(template="plotly_dark", height=220,
                                         paper_bgcolor="#1e293b", margin=dict(t=30, b=5, l=5, r=5))
                    st.plotly_chart(fig_ch, use_container_width=True)
                    st.markdown(f"Mean: {img[:,:,ch].mean():.0f} | Std: {img[:,:,ch].std():.0f}")

            info("The CNN learns which pixel combinations (at which spatial positions) correlate with defects — no human-designed features needed. That's the power of deep learning on images.")

        else:  # Augmentation
            st.markdown("### Data Augmentation — Multiply Your Training Data")
            st.markdown("""
You have 200 defect images. You need 5,000 to train a robust CNN.
Data augmentation creates new training examples by applying **label-preserving transformations**.
            """)
            augmentations = pd.DataFrame({
                "Technique": ["Horizontal flip", "Vertical flip", "Rotation ±15°", "Brightness ±20%",
                              "Gaussian noise", "Crop + resize", "Cutout (random erase)", "Mixup"],
                "Effect": ["Left-right invariance", "Top-bottom invariance", "Orientation invariance",
                           "Lighting invariance", "Sensor noise robustness", "Scale invariance",
                           "Occlusion robustness", "Soft label regularisation"],
                "Effective for": ["Most images", "Satellite / top-down", "Rotated parts",
                                  "Variable lighting", "Noisy sensors", "Varying distances",
                                  "Partially hidden defects", "Small datasets"],
                "PyTorch code": [
                    "T.RandomHorizontalFlip(p=0.5)",
                    "T.RandomVerticalFlip(p=0.5)",
                    "T.RandomRotation(15)",
                    "T.ColorJitter(brightness=0.2)",
                    "T.GaussianBlur(3)",
                    "T.RandomResizedCrop(224)",
                    "T.RandomErasing(p=0.5)",
                    "# custom",
                ],
            })
            st.dataframe(augmentations, use_container_width=True, hide_index=True)

            st.markdown("### Multiplier Effect")
            n_orig = st.slider("Original images per class:", 50, 500, 100)
            n_augs = st.multiselect("Augmentations to apply:", augmentations["Technique"].tolist(),
                                    default=["Horizontal flip", "Rotation ±15°", "Brightness ±20%", "Crop + resize"])
            effective = n_orig * max(1, len(n_augs)) * 3
            c1, c2, c3 = st.columns(3)
            c1.metric("Original images", n_orig)
            c2.metric("Augmentations applied", len(n_augs))
            c3.metric("Effective training size", f"~{effective:,}")
            success(f"With {len(n_augs)} augmentation techniques, your {n_orig} images become ~{effective:,} effective training examples — often enough to fine-tune a ResNet.")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 10 — QUIZ
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[9]:
        st.markdown("### Quiz — AI Applications (10 questions)")
        questions = [
            {"q": "Which neural network architecture is best for image-based defect detection?",
             "options": ["RNN", "CNN", "Transformer", "Linear Regression"],
             "answer": "CNN",
             "explanation": "CNNs learn spatial features in images — edges, textures, shapes — through convolutional filters. RNNs are for sequences, Transformers for language."},
            {"q": "What does RAG stand for?",
             "options": ["Random Augmented Generation", "Retrieval Augmented Generation", "Recurrent Attention Graph", "Recursive AI Generation"],
             "answer": "Retrieval Augmented Generation",
             "explanation": "RAG retrieves relevant documents and injects them into the LLM prompt — grounding answers in your actual documentation instead of the model's (potentially outdated) training data."},
            {"q": "Isolation Forest detects anomalies by:",
             "options": ["Measuring distance from cluster centroids", "Isolating points that require fewer random splits", "Computing z-scores", "Training on labelled failure data"],
             "answer": "Isolating points that require fewer random splits",
             "explanation": "Anomalies are rare and different from normal points. Random tree splits isolate them faster (fewer splits). Short average path length = anomaly."},
            {"q": "What is the key advantage of DBSCAN over K-Means?",
             "options": ["Always faster", "Finds clusters of arbitrary shape and marks outliers as noise", "Requires K to be specified", "Works better on high-dimensional data"],
             "answer": "Finds clusters of arbitrary shape and marks outliers as noise",
             "explanation": "K-Means assumes spherical clusters and forces every point into a cluster. DBSCAN discovers arbitrary shapes and explicitly marks outliers — critical for fault mode discovery."},
            {"q": "In PCA, the first principal component is:",
             "options": ["The feature with highest variance", "The direction of maximum variance in the data", "The most correlated pair of features", "The mean of all features"],
             "answer": "The direction of maximum variance in the data",
             "explanation": "PCA finds orthogonal directions (principal components) of maximum variance. PC1 captures the most variance, PC2 the next most, etc. It's a direction in feature space, not a single feature."},
            {"q": "TF-IDF downweights words that appear in many documents because:",
             "options": ["They are grammatically incorrect", "They carry less discriminative information", "They are too long", "They are stop words"],
             "answer": "They carry less discriminative information",
             "explanation": "Words like 'the', 'engine', 'is' appear in every document — they don't help distinguish one document from another. TF-IDF multiplies term frequency by inverse document frequency to penalise common words."},
            {"q": "What is the 'vanishing gradient' problem in RNNs?",
             "options": ["Memory runs out during training", "Gradients become too small to update weights for early time steps", "The model forgets to use the recurrent connection", "Batch size is too small"],
             "answer": "Gradients become too small to update weights for early time steps",
             "explanation": "In backpropagation through time, gradients are multiplied at each step. With many steps they shrink to near-zero, so early steps don't get updated — the model can't learn long-term dependencies. LSTMs solve this with gating."},
            {"q": "Fine-tuning a pre-trained CNN (transfer learning) is preferred over training from scratch because:",
             "options": ["Pre-trained CNNs have no weights", "It reuses learned features (edges, textures) and needs less data and time", "CNNs can't be trained from scratch", "Pre-trained models have better architectures"],
             "answer": "It reuses learned features (edges, textures) and needs less data and time",
             "explanation": "A CNN pre-trained on ImageNet already knows how to detect edges, textures, and shapes. You only need to re-train the final classification head on your specific defect classes — 100× less data and time."},
            {"q": "Which LLM limitation means you MUST use RAG for up-to-date information?",
             "options": ["Hallucination", "Context window", "Knowledge cutoff", "Bias"],
             "answer": "Knowledge cutoff",
             "explanation": "LLMs are trained on data up to a fixed date. They cannot know about recent incidents, updated manuals, or your latest product versions. RAG retrieves current documents at query time."},
            {"q": "Data augmentation in CV is used to:",
             "options": ["Remove noisy images", "Increase effective training set size with label-preserving transforms", "Reduce model size", "Speed up inference"],
             "answer": "Increase effective training set size with label-preserving transforms",
             "explanation": "Rotating, flipping, cropping, and adjusting brightness on defect images creates new training examples. A defect remains a defect under these transforms — so you get more data for free."},
        ]

        score = 0
        answered = 0
        for i, q in enumerate(questions):
            st.markdown(f"**Q{i+1}.** {q['q']}")
            choice = st.radio("", q["options"], key=f"quiz_09_q{i}", index=None, horizontal=True)
            if choice:
                answered += 1
                if choice == q["answer"]:
                    st.markdown(f'<div class="box-success">✅ Correct! {q["explanation"]}</div>', unsafe_allow_html=True)
                    score += 1
                else:
                    st.markdown(f'<div class="box-warning">❌ Correct answer: <strong>{q["answer"]}</strong>. {q["explanation"]}</div>', unsafe_allow_html=True)
            st.markdown("")

        if answered == len(questions):
            pct = int(score / len(questions) * 100)
            st.markdown("---")
            if pct == 100:
                st.balloons()
                success(f"🏆 Perfect score! {score}/{len(questions)} — You're ready to build production AI systems!")
            elif pct >= 70:
                success(f"✅ Strong result: {score}/{len(questions)} ({pct}%) — Solid AI foundations!")
            else:
                warn(f"📚 {score}/{len(questions)} ({pct}%) — Review the tabs above and try again.")

            if "ai_applications" not in st.session_state.completed_topics:
                st.session_state.completed_topics.append("ai_applications")
                st.session_state.quiz_score += score
                st.session_state.quiz_total += len(questions)
