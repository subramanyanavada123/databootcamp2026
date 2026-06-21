import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

DARK_TEMPLATE = "plotly_dark"
COLORS = px.colors.qualitative.Set2


def fig_defaults(fig, title="", height=400):
    fig.update_layout(
        template=DARK_TEMPLATE,
        height=height,
        title=dict(text=title, font=dict(size=14, color="#f1f5f9")),
        paper_bgcolor="#1e293b",
        plot_bgcolor="#0f172a",
        font=dict(color="#94a3b8"),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def plot_distribution(df: pd.DataFrame, col: str, color_col: str = None):
    if color_col and color_col in df.columns:
        fig = px.histogram(df, x=col, color=color_col, nbins=40,
                           color_discrete_sequence=COLORS, barmode="overlay", opacity=0.8)
    else:
        fig = px.histogram(df, x=col, nbins=40, color_discrete_sequence=["#3b82f6"])
    return fig_defaults(fig, f"Distribution of {col}")


def plot_scatter(df: pd.DataFrame, x: str, y: str, color: str = None, size: str = None):
    fig = px.scatter(df, x=x, y=y, color=color, size=size,
                     color_discrete_sequence=COLORS, opacity=0.7)
    return fig_defaults(fig, f"{y} vs {x}", height=450)


def plot_correlation_heatmap(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    corr = df[num_cols].corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale="RdBu",
        zmid=0,
        text=corr.values.round(2),
        texttemplate="%{text}",
        showscale=True,
    ))
    return fig_defaults(fig, "Correlation Matrix", height=500)


def plot_timeseries(df: pd.DataFrame, time_col: str, value_cols: list):
    fig = go.Figure()
    for i, col in enumerate(value_cols):
        fig.add_trace(go.Scatter(
            x=df[time_col], y=df[col], name=col,
            line=dict(color=COLORS[i % len(COLORS)], width=1.5),
            mode="lines",
        ))
    return fig_defaults(fig, "Time Series", height=400)


def plot_box(df: pd.DataFrame, x: str, y: str):
    fig = px.box(df, x=x, y=y, color=x, color_discrete_sequence=COLORS)
    return fig_defaults(fig, f"{y} by {x}")


def plot_bar(df: pd.DataFrame, x: str, y: str, color: str = None):
    fig = px.bar(df, x=x, y=y, color=color or x,
                 color_discrete_sequence=COLORS, barmode="group")
    return fig_defaults(fig, f"{y} by {x}")


def plot_missing_heatmap(df: pd.DataFrame):
    miss = df.isnull().astype(int)
    if miss.sum().sum() == 0:
        return None
    fig = go.Figure(go.Heatmap(
        z=miss.T.values,
        x=[str(i) for i in miss.index],
        y=miss.columns.tolist(),
        colorscale=[[0, "#1e3a5f"], [1, "#ef4444"]],
        showscale=False,
    ))
    fig.update_xaxes(showticklabels=False)
    return fig_defaults(fig, "Missing Values (red = missing)", height=300)


def plot_feature_importance(feature_names: list, importances: list):
    pairs = sorted(zip(importances, feature_names), reverse=True)
    imp, names = zip(*pairs)
    fig = go.Figure(go.Bar(
        x=list(imp), y=list(names),
        orientation="h",
        marker_color="#3b82f6",
    ))
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return fig_defaults(fig, "Feature Importance", height=max(300, len(names) * 30))


def plot_confusion_matrix(cm: np.ndarray, labels: list):
    fig = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale="Blues",
        text=cm, texttemplate="%{text}",
        showscale=True,
    ))
    return fig_defaults(fig, "Confusion Matrix", height=400)


def plot_roc_curve(fpr, tpr, auc_score: float):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"ROC (AUC={auc_score:.3f})",
                             line=dict(color="#3b82f6", width=2)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random",
                             line=dict(color="#64748b", dash="dash")))
    fig.update_xaxes(title="False Positive Rate")
    fig.update_yaxes(title="True Positive Rate")
    return fig_defaults(fig, "ROC Curve", height=400)


def plot_learning_curve(train_sizes, train_scores, val_scores):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_sizes, y=train_scores.mean(axis=1),
                             name="Train", line=dict(color="#10b981", width=2)))
    fig.add_trace(go.Scatter(x=train_sizes, y=val_scores.mean(axis=1),
                             name="Validation", line=dict(color="#f59e0b", width=2)))
    fig.update_xaxes(title="Training Size")
    fig.update_yaxes(title="Score")
    return fig_defaults(fig, "Learning Curve", height=400)


def plot_pca_2d(X_pca: np.ndarray, labels=None, label_name="Label"):
    df_pca = pd.DataFrame({"PC1": X_pca[:, 0], "PC2": X_pca[:, 1]})
    if labels is not None:
        df_pca[label_name] = labels.astype(str)
        fig = px.scatter(df_pca, x="PC1", y="PC2", color=label_name,
                         color_discrete_sequence=COLORS, opacity=0.8)
    else:
        fig = px.scatter(df_pca, x="PC1", y="PC2",
                         color_discrete_sequence=["#3b82f6"], opacity=0.8)
    return fig_defaults(fig, "PCA — 2D Projection", height=450)
