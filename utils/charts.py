"""
Chart builder for the Data Q&A tab.

Keeps chart logic separate from app.py so the UI file stays readable.
Uses Plotly for interactive (zoomable, hoverable) charts.
"""
import pandas as pd
import plotly.express as px

CHART_TYPES = ["Bar", "Line", "Scatter", "Pie", "Histogram", "Box"]


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include="number").columns.tolist()


def categorical_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(exclude="number").columns.tolist()


def build_chart(df: pd.DataFrame, chart_type: str, x: str, y: str | None = None, color: str | None = None):
    """Returns a Plotly figure, or raises ValueError with a friendly message on bad input."""
    color_arg = color if color and color != "(none)" else None

    if chart_type == "Bar":
        if not y:
            raise ValueError("Bar charts need a Y-axis column.")
        fig = px.bar(df, x=x, y=y, color=color_arg, barmode="group")
    elif chart_type == "Line":
        if not y:
            raise ValueError("Line charts need a Y-axis column.")
        fig = px.line(df, x=x, y=y, color=color_arg, markers=True)
    elif chart_type == "Scatter":
        if not y:
            raise ValueError("Scatter charts need a Y-axis column.")
        fig = px.scatter(df, x=x, y=y, color=color_arg)
    elif chart_type == "Pie":
        if not y:
            raise ValueError("Pie charts need a values column (Y-axis).")
        fig = px.pie(df, names=x, values=y)
    elif chart_type == "Histogram":
        fig = px.histogram(df, x=x, color=color_arg)
    elif chart_type == "Box":
        fig = px.box(df, x=x, y=y, color=color_arg)
    else:
        raise ValueError(f"Unknown chart type: {chart_type}")

    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=450)
    return fig
