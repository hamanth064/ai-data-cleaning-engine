import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Consistent dark-themed color palette
PALETTE = px.colors.qualitative.Vivid
TEMPLATE = "plotly_dark"
BG_COLOR = "#0f1117"
PAPER_COLOR = "#1a1d27"


def _base_layout(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#e2e8f0"), x=0.02),
        paper_bgcolor=PAPER_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color="#94a3b8", family="'DM Sans', sans-serif"),
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.1)"),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)")
    return fig


def get_numeric_distributions(df: pd.DataFrame, max_cols: int = 8) -> list:
    """Returns histogram figures for each numeric column."""
    charts = []
    numeric_cols = df.select_dtypes(include="number").columns[:max_cols]

    for i, col in enumerate(numeric_cols):
        fig = px.histogram(
            df,
            x=col,
            nbins=40,
            color_discrete_sequence=[PALETTE[i % len(PALETTE)]],
            template=TEMPLATE,
            marginal="box",
        )
        fig = _base_layout(fig, f"Distribution — {col}")
        charts.append(fig)

    return charts


def get_categorical_distributions(df: pd.DataFrame, max_cols: int = 6) -> list:
    """Returns bar chart figures for top categories in each categorical column."""
    charts = []
    cat_cols = df.select_dtypes(include=["object", "category"]).columns[:max_cols]

    for i, col in enumerate(cat_cols):
        top = df[col].value_counts().head(15).reset_index()
        top.columns = [col, "count"]

        fig = px.bar(
            top,
            x=col,
            y="count",
            color="count",
            color_continuous_scale="Viridis",
            template=TEMPLATE,
        )
        fig.update_coloraxes(showscale=False)
        fig = _base_layout(fig, f"Top Categories — {col}")
        charts.append(fig)

    return charts


def get_correlation_heatmap(df: pd.DataFrame):
    """Returns a correlation heatmap for numeric columns. None if < 2 numeric cols."""
    numeric_df = df.select_dtypes(include="number")
    if len(numeric_df.columns) < 2:
        return None

    corr = numeric_df.corr().round(2)

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.columns.tolist(),
            colorscale="RdBu",
            zmid=0,
            text=corr.values.round(2),
            texttemplate="%{text}",
            textfont=dict(size=11),
            hoverongaps=False,
        )
    )
    fig = _base_layout(fig, "Correlation Matrix")
    fig.update_layout(height=500)
    return fig


def get_missing_values_bar(df: pd.DataFrame):
    """Returns a horizontal bar chart of missing value % per column."""
    missing = (df.isnull().sum() / len(df) * 100).sort_values(ascending=True)
    missing = missing[missing > 0]

    if missing.empty:
        return None

    fig = px.bar(
        x=missing.values,
        y=missing.index,
        orientation="h",
        labels={"x": "Missing %", "y": "Column"},
        color=missing.values,
        color_continuous_scale="OrRd",
        template=TEMPLATE,
    )
    fig.update_coloraxes(showscale=False)
    fig = _base_layout(fig, "Missing Values by Column (%)")
    fig.update_layout(height=max(250, len(missing) * 35))
    return fig


def get_dtypes_pie(df: pd.DataFrame):
    """Returns a donut chart of column data types."""
    dtype_counts = df.dtypes.astype(str).value_counts().reset_index()
    dtype_counts.columns = ["dtype", "count"]

    fig = px.pie(
        dtype_counts,
        names="dtype",
        values="count",
        hole=0.55,
        color_discrete_sequence=PALETTE,
        template=TEMPLATE,
    )
    fig = _base_layout(fig, "Column Data Types")
    fig.update_traces(textposition="outside", textinfo="percent+label")
    return fig
