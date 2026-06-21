import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from app.utils.datasets import DATASETS
from app.utils.notebook_export import make_notebook, TOPIC_NOTEBOOKS
from app.utils.visualizations import (
    plot_confusion_matrix, plot_roc_curve,
    plot_feature_importance, plot_learning_curve, fig_defaults
)
import plotly.express as px
import plotly.graph_objects as go

def render():
    st.markdown("""
    <div class="hero-banner" style="padding:1.5rem;">
      <h1 style="font-size:1.8rem;">🤖 ML Fundamentals</h1>
      <p>Teaching machines to learn from data — training, evaluation, and model selection</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📖 Explain", "🎯 Train a Model", "💻 Code", "✅ Quiz"])

    with tab1:
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown("### What is Machine Learning?")
            st.markdown("""
Traditional programming: **Rules + Data → Output**

Machine Learning: **Data + Output → Rules (the model)**

The model *learns* patterns from historical data and applies them to new, unseen data.
            """)
            st.code("""
THE ML LEARNING LOOP
────────────────────────────────────────────────────────────────
      Training Data (X, y)
             │
             ▼
   ┌─── Algorithm ────┐
   │  1. Initialize   │
   │  2. Make guess   │
   │  3. Measure loss │
   │  4. Adjust       │
   │  5. Repeat       │
   └──────────────────┘
             │
             ▼
         Model (learned parameters)
             │
             ▼
    New data X_new → Predictions ŷ
            """, language="text")

            st.markdown("### ML Types")
            st.code("""
SUPERVISED         UNSUPERVISED        REINFORCEMENT
─────────────────  ──────────────────  ─────────────────
Has labels (y)     No labels           Agent + Environment

Classification:    Clustering:         Game AI
  Spam or not?       K-Means           Robot control
  Failure? Yes/No    DBSCAN            AlphaGo
  Digit: 0-9         Hierarchical

Regression:        Dimensionality:
  Predict price      PCA
  Predict RPM        t-SNE
  Predict demand     UMAP

                   Anomaly Detection:
                     Isolation Forest
                     Autoencoder
            """, language="text")

        with col2:
            st.markdown("### Model Comparison")
            models = {
                "Model": ["Linear Regression", "Logistic Reg.", "Decision Tree", "Random Forest", "XGBoost", "SVM", "Neural Network", "KNN"],
                "Type": ["Regression", "Classification", "Both", "Both", "Both", "Both", "Both", "Both"],
                "Speed": ["⚡⚡⚡", "⚡⚡⚡", "⚡⚡⚡", "⚡⚡", "⚡⚡", "⚡", "⚡", "⚡"],
                "Accuracy": ["★★", "★★", "★★", "★★★★", "★★★★★", "★★★", "★★★★★", "★★★"],
                "Interpretable": ["✅", "✅", "✅", "⚠️", "⚠️", "❌", "❌", "⚠️"],
            }
            st.dataframe(pd.DataFrame(models), use_container_width=True, hide_index=True)

            st.markdown("### Evaluation Metrics")
            metrics = {
                "Problem": ["Classification", "Classification", "Classification", "Classification", "Regression", "Regression", "Regression"],
                "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "MAE", "RMSE", "R²"],
                "Formula": ["TP+TN/total", "TP/(TP+FP)", "TP/(TP+FN)", "2·P·R/(P+R)", "mean|y-ŷ|", "√mean(y-ŷ)²", "1-SS_res/SS_tot"],
            }
            st.dataframe(pd.DataFrame(metrics), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("""
        <div class="box-info">
        <strong>🧠 AI Perspective:</strong>
        The difference between ML and traditional AI: ML models improve with MORE data.
        GPT-4 is a transformer neural network trained on ~1 trillion tokens.
        Random Forest with 100 trees on 10,000 rows can outperform a neural network on tabular data.
        Always start simple. Complex ≠ better.
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("### Live Model Training Lab")

        ds_key = st.selectbox("Dataset:", [
            "🔩 Manufacturing QC (Industry)",
            "📱 Telecom Churn (Telco)",
            "🏥 Patient Readmission (Healthcare)",
        ])
        df = DATASETS[ds_key]()

        # Auto-detect target
        target_map = {
            "🔩 Manufacturing QC (Industry)": "passed_qc",
            "📱 Telecom Churn (Telco)": "churned",
            "🏥 Patient Readmission (Healthcare)": "readmitted",
        }
        target = target_map[ds_key]
        st.markdown(f"**Target variable:** `{target}`  |  **Positive class:** True")

        # Feature selection
        exclude = [target, "part_id", "customer_id", "patient_id"]
        num_cols = [c for c in df.select_dtypes(include=np.number).columns if c not in exclude]
        cat_cols = [c for c in df.select_dtypes(include="object").columns if c not in exclude]

        selected_features = st.multiselect("Select features:", num_cols + cat_cols,
                                           default=num_cols[:4])

        col1, col2 = st.columns(2)
        model_type = col1.selectbox("Algorithm:", [
            "Logistic Regression", "Random Forest", "XGBoost", "Decision Tree", "KNN"
        ])
        test_size = col2.slider("Test set size:", 0.1, 0.4, 0.2)

        if st.button("🚀 Train Model", type="primary") and selected_features:
            from sklearn.model_selection import train_test_split, learning_curve
            from sklearn.preprocessing import StandardScaler, LabelEncoder
            from sklearn.impute import SimpleImputer
            from sklearn.metrics import (accuracy_score, classification_report,
                                         confusion_matrix, roc_auc_score, roc_curve)
            from sklearn.pipeline import Pipeline
            from sklearn.compose import ColumnTransformer
            from sklearn.preprocessing import OneHotEncoder

            df_ml = df[selected_features + [target]].copy()
            df_ml[target] = df_ml[target].astype(int)

            sel_num = [c for c in selected_features if c in num_cols]
            sel_cat = [c for c in selected_features if c in cat_cols]

            transformers = []
            if sel_num:
                transformers.append(("num", Pipeline([
                    ("imp", SimpleImputer(strategy="median")),
                    ("sc",  StandardScaler()),
                ]), sel_num))
            if sel_cat:
                transformers.append(("cat", Pipeline([
                    ("imp", SimpleImputer(strategy="most_frequent")),
                    ("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]), sel_cat))

            preprocessor = ColumnTransformer(transformers)

            if model_type == "Logistic Regression":
                from sklearn.linear_model import LogisticRegression
                clf = LogisticRegression(max_iter=500, random_state=42)
            elif model_type == "Random Forest":
                from sklearn.ensemble import RandomForestClassifier
                clf = RandomForestClassifier(n_estimators=100, random_state=42)
            elif model_type == "XGBoost":
                from sklearn.ensemble import GradientBoostingClassifier
                clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
            elif model_type == "Decision Tree":
                from sklearn.tree import DecisionTreeClassifier
                clf = DecisionTreeClassifier(max_depth=5, random_state=42)
            else:
                from sklearn.neighbors import KNeighborsClassifier
                clf = KNeighborsClassifier(n_neighbors=5)

            model = Pipeline([("pre", preprocessor), ("clf", clf)])
            X = df_ml[selected_features]
            y = df_ml[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                                  random_state=42, stratify=y)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            acc   = accuracy_score(y_test, y_pred)
            auc   = roc_auc_score(y_test, y_prob)
            cm    = confusion_matrix(y_test, y_pred)
            fpr, tpr, _ = roc_curve(y_test, y_prob)

            st.markdown("### Results")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy",       f"{acc*100:.1f}%")
            c2.metric("ROC AUC",        f"{auc:.3f}")
            c3.metric("Train samples",  f"{len(X_train):,}")
            c4.metric("Test samples",   f"{len(X_test):,}")

            col_cm, col_roc = st.columns(2)
            with col_cm:
                st.plotly_chart(plot_confusion_matrix(cm, ["Negative", "Positive"]),
                                use_container_width=True)
            with col_roc:
                st.plotly_chart(plot_roc_curve(fpr, tpr, auc),
                                use_container_width=True)

            # Feature importance (RF only)
            if model_type in ["Random Forest", "XGBoost", "Decision Tree"]:
                try:
                    pre = model.named_steps["pre"]
                    imp = model.named_steps["clf"].feature_importances_
                    feat_names = []
                    if sel_num:
                        feat_names += sel_num
                    if sel_cat:
                        enc = pre.named_transformers_["cat"].named_steps["enc"]
                        feat_names += enc.get_feature_names_out(sel_cat).tolist()
                    if len(feat_names) == len(imp):
                        st.plotly_chart(plot_feature_importance(feat_names, imp.tolist()),
                                        use_container_width=True)
                except Exception:
                    pass

            st.markdown("#### Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

    with tab3:
        st.markdown("### ML Code — End to End")
        nb_bytes = make_notebook("ML Fundamentals — DS Workshop 2026", TOPIC_NOTEBOOKS.get("ML Fundamentals", []))
        st.download_button("📓 Download as Jupyter Notebook", nb_bytes,
                           file_name="08_ml_fundamentals.ipynb", mime="application/json")
        st.markdown("---")
                st.code("""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# ── 1. Load and prepare data ──────────────────────────────────────────
df = pd.read_parquet("processed_data.parquet")
X = df.drop("churned", axis=1)
y = df["churned"].astype(int)

numeric_features    = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include="object").columns.tolist()

# ── 2. Preprocessing pipeline ────────────────────────────────────────
preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ]), numeric_features),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), categorical_features),
])

# ── 3. Full pipeline with model ───────────────────────────────────────
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier",   RandomForestClassifier(n_estimators=100, random_state=42)),
])

# ── 4. Train / test split ────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model.fit(X_train, y_train)

# ── 5. Evaluation ────────────────────────────────────────────────────
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.3f}")

# ── 6. Cross-validation (more robust estimate) ───────────────────────
cv_scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
print(f"CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# ── 7. Hyperparameter tuning ─────────────────────────────────────────
param_grid = {
    "classifier__n_estimators": [50, 100, 200],
    "classifier__max_depth":    [5, 10, None],
}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
print(f"Best AUC: {grid_search.best_score_:.3f}")

# ── 8. Save the model ────────────────────────────────────────────────
import joblib
joblib.dump(grid_search.best_estimator_, "churn_model.pkl")
        """, language="python")

    with tab4:
        st.markdown("### Quiz — ML Fundamentals")
        questions = [
            {"q": "Your model has 99% training accuracy and 55% test accuracy. What's the problem?",
             "options": ["Underfitting", "Overfitting", "Data leakage", "Wrong metric"],
             "answer": "Overfitting",
             "explanation": "Huge train/test gap = overfitting. The model memorized training data instead of learning general patterns."},
            {"q": "Why should you use stratified train/test split for imbalanced datasets?",
             "options": ["It's faster", "Preserves class proportions in both sets", "Prevents overfitting", "Required by sklearn"],
             "answer": "Preserves class proportions in both sets",
             "explanation": "Without stratification, your test set might have 0 fraud cases in a 1% fraud dataset. Stratified split ensures both sets reflect the real distribution."},
            {"q": "ROC AUC = 0.5 means what?",
             "options": ["50% accuracy", "Model performs at random chance", "50% precision", "Model needs more data"],
             "answer": "Model performs at random chance",
             "explanation": "AUC of 0.5 = random guessing. A perfect model = 1.0. AUC < 0.5 means predictions are systematically inverted."},
        ]
        for i, q in enumerate(questions):
            st.markdown(f"**Q{i+1}.** {q['q']}")
            choice = st.radio("", q["options"], key=f"quiz_08_q{i}", index=None, horizontal=True)
            if choice:
                if choice == q["answer"]:
                    st.markdown(f'<div class="box-success">✅ Correct! {q["explanation"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="box-warning">❌ Answer: **{q["answer"]}**. {q["explanation"]}</div>', unsafe_allow_html=True)
            st.markdown("")
        if "ml_fundamentals" not in st.session_state.completed_topics:
            if st.button("✅ Mark topic as complete"):
                st.session_state.completed_topics.append("ml_fundamentals")
                st.success("Topic marked complete!")
