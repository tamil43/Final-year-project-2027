"""
Plotly Visualizations & Interactive Dashboard Charts.
Provides professional charts for:
  - Forecast Comparison (Demand vs Supply)
  - Probabilistic Prediction Intervals (90%, 95%, 99% confidence bands)
  - Demand–Supply Gap Probabilistic Distribution
  - Risk Classification Gauge
  - Historical Validation Benchmarks
"""

from typing import Dict, Any, List
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
import config


# Consistent Dashboard Color Palette
COLOR_DEMAND = "#3B82F6"      # Indigo / Blue
COLOR_SUPPLY = "#10B981"      # Emerald Green
COLOR_GAP = "#8B5CF6"         # Purple / Violet
COLOR_SHORTAGE = "#EF4444"    # Red
COLOR_SURPLUS = "#059669"     # Forest Green
BG_COLOR = "rgba(0,0,0,0)"
GRID_COLOR = "rgba(150, 150, 150, 0.15)"


def create_forecast_comparison_chart(
    predicted_demand: float,
    predicted_supply: float,
    predicted_gap: float,
    month_label: str = "Forecast Period",
) -> go.Figure:
    """
    Creates an interactive bar chart comparing Predicted Demand, Predicted Supply, and Gap.
    """
    categories = ["Predicted Demand", "Predicted Supply", "Demand–Supply Gap"]
    values = [predicted_demand, predicted_supply, predicted_gap]
    colors = [COLOR_DEMAND, COLOR_SUPPLY, COLOR_SHORTAGE if predicted_gap > 0 else COLOR_SURPLUS]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=categories,
            y=values,
            text=[f"{v:,.2f} MU" for v in values],
            textposition="auto",
            marker=dict(
                color=colors,
                line=dict(color="rgba(255,255,255,0.2)", width=1.5),
            ),
            hovertemplate="<b>%{x}</b><br>Value: %{y:,.2f} MU<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text=f"<b>Electricity Demand vs Supply Comparison</b> ({month_label})",
            font=dict(size=18),
        ),
        yaxis=dict(
            title="Electricity Quantity (MU)",
            gridcolor=GRID_COLOR,
            zeroline=True,
            zerolinecolor="rgba(200,200,200,0.4)",
        ),
        xaxis=dict(gridcolor=GRID_COLOR),
        plot_bgcolor=BG_COLOR,
        paper_bgcolor=BG_COLOR,
        margin=dict(l=40, r=40, t=60, b=40),
        height=380,
    )
    return fig


def create_prediction_interval_chart(
    metric_name: str,
    interval_data: Dict[str, Any],
    theme_color: str = "#3B82F6",
) -> go.Figure:
    """
    Creates a detailed horizontal interval chart displaying point forecast
    and 90%, 95%, 99% symmetric confidence bounds.
    """
    point = interval_data["point_forecast"]
    pi_90 = interval_data["pi_90"]
    pi_95 = interval_data["pi_95"]
    pi_99 = interval_data["pi_99"]

    fig = go.Figure()

    # 99% Interval
    fig.add_trace(
        go.Scatter(
            x=[pi_99["lower"], pi_99["upper"]],
            y=["99% Interval", "99% Interval"],
            mode="lines+markers",
            name="99% Interval",
            line=dict(color="#A855F7", width=6),
            marker=dict(size=10, symbol="line-ns-open"),
            hovertemplate=f"99% Range: [{pi_99['lower']:,.2f}, {pi_99['upper']:,.2f}] MU<extra></extra>",
        )
    )

    # 95% Interval
    fig.add_trace(
        go.Scatter(
            x=[pi_95["lower"], pi_95["upper"]],
            y=["95% Interval", "95% Interval"],
            mode="lines+markers",
            name="95% Interval",
            line=dict(color="#3B82F6", width=8),
            marker=dict(size=12, symbol="line-ns-open"),
            hovertemplate=f"95% Range: [{pi_95['lower']:,.2f}, {pi_95['upper']:,.2f}] MU<extra></extra>",
        )
    )

    # 90% Interval
    fig.add_trace(
        go.Scatter(
            x=[pi_90["lower"], pi_90["upper"]],
            y=["90% Interval", "90% Interval"],
            mode="lines+markers",
            name="90% Interval",
            line=dict(color="#10B981", width=10),
            marker=dict(size=14, symbol="line-ns-open"),
            hovertemplate=f"90% Range: [{pi_90['lower']:,.2f}, {pi_90['upper']:,.2f}] MU<extra></extra>",
        )
    )

    # Point Forecast Marker (Vertical Reference across all tiers)
    for y_cat in ["90% Interval", "95% Interval", "99% Interval"]:
        fig.add_trace(
            go.Scatter(
                x=[point],
                y=[y_cat],
                mode="markers",
                name="Point Forecast",
                showlegend=(y_cat == "90% Interval"),
                marker=dict(color="#F59E0B", size=14, symbol="diamond"),
                hovertemplate=f"Point Forecast: {point:,.2f} MU<extra></extra>",
            )
        )

    # Vertical line at point forecast
    fig.add_vline(
        x=point,
        line_width=1.5,
        line_dash="dash",
        line_color="#F59E0B",
        annotation_text=f"Point: {point:,.2f} MU",
        annotation_position="top right",
    )

    fig.update_layout(
        title=dict(
            text=f"<b>{metric_name} - Probabilistic Prediction Intervals</b> (σ = {interval_data['sigma']:.2f} MU)",
            font=dict(size=16),
        ),
        xaxis=dict(
            title="Electricity (MU)",
            gridcolor=GRID_COLOR,
        ),
        yaxis=dict(gridcolor=GRID_COLOR),
        plot_bgcolor=BG_COLOR,
        paper_bgcolor=BG_COLOR,
        margin=dict(l=40, r=40, t=60, b=40),
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def create_gap_distribution_chart(
    predicted_gap: float,
    gap_sigma: float,
    gap_intervals: Dict[str, Any],
) -> go.Figure:
    """
    Creates the normal probability density function (PDF) curve for the forecasted gap
    highlighting the 0 MU balance line, risk regions, and confidence intervals.
    """
    x_min = predicted_gap - 3.5 * gap_sigma
    x_max = predicted_gap + 3.5 * gap_sigma
    x = np.linspace(x_min, x_max, 300)
    pdf = norm.pdf(x, loc=predicted_gap, scale=gap_sigma)

    pi_95 = gap_intervals["pi_95"]
    x_95 = np.linspace(pi_95["lower"], pi_95["upper"], 150)
    pdf_95 = norm.pdf(x_95, loc=predicted_gap, scale=gap_sigma)

    fig = go.Figure()

    # Base PDF curve
    fig.add_trace(
        go.Scatter(
            x=x,
            y=pdf,
            mode="lines",
            name="Gap Probability Density",
            line=dict(color="#8B5CF6", width=3),
            hovertemplate="Gap: %{x:,.2f} MU<br>Density: %{y:.6f}<extra></extra>",
        )
    )

    # 95% Confidence shaded region
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([x_95, x_95[::-1]]),
            y=np.concatenate([pdf_95, np.zeros_like(pdf_95)]),
            fill="toself",
            fillcolor="rgba(139, 92, 246, 0.25)",
            line=dict(color="rgba(255,255,255,0)"),
            name="95% Confidence Interval",
            hoverinfo="skip",
        )
    )

    # Mean / Point Gap vertical line
    fig.add_vline(
        x=predicted_gap,
        line_width=2,
        line_color="#F59E0B",
        annotation_text=f"Expected Gap: {predicted_gap:,.2f} MU",
        annotation_position="top left",
    )

    # 0 MU Zero-Gap Balance Reference Line
    fig.add_vline(
        x=0,
        line_width=2,
        line_dash="dot",
        line_color="#10B981",
        annotation_text="Equilibrium (0 MU)",
        annotation_position="bottom right",
    )

    fig.update_layout(
        title=dict(
            text=f"<b>Probabilistic Distribution of Demand–Supply Gap</b> (σ_Gap = {gap_sigma:,.2f} MU)",
            font=dict(size=16),
        ),
        xaxis=dict(
            title="Forecasted Gap (MU)  [Positive = Shortage | Negative = Surplus]",
            gridcolor=GRID_COLOR,
        ),
        yaxis=dict(
            title="Probability Density",
            gridcolor=GRID_COLOR,
            showticklabels=False,
        ),
        plot_bgcolor=BG_COLOR,
        paper_bgcolor=BG_COLOR,
        margin=dict(l=40, r=40, t=60, b=40),
        height=340,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def create_risk_gauge(predicted_gap: float, risk_level: str) -> go.Figure:
    """
    Creates an interactive gauge meter showing the position of the forecasted gap
    relative to project-defined risk thresholds (0 - 3000 Low, 3000 - 4500 Moderate, >4500 High).
    """
    max_gauge = max(6000.0, predicted_gap * 1.25)
    gauge_val = max(0.0, predicted_gap)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=gauge_val,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": f"<b>Risk Classification Gauge</b><br><span style='font-size:13px;color:gray'>Risk: {risk_level.upper()}</span>", "font": {"size": 17}},
            delta={"reference": 3000, "increasing": {"color": "#EF4444"}, "decreasing": {"color": "#10B981"}},
            gauge={
                "axis": {"range": [0, max_gauge], "tickwidth": 1, "tickcolor": "gray"},
                "bar": {"color": "#3B82F6", "thickness": 0.3},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 1,
                "bordercolor": "gray",
                "steps": [
                    {"range": [0, 3000], "color": "rgba(59, 130, 246, 0.25)"},
                    {"range": [3000, 4500], "color": "rgba(245, 158, 11, 0.3)"},
                    {"range": [4500, max_gauge], "color": "rgba(239, 68, 68, 0.35)"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 4500,
                },
            },
        )
    )

    fig.update_layout(
        plot_bgcolor=BG_COLOR,
        paper_bgcolor=BG_COLOR,
        height=300,
        margin=dict(l=30, r=30, t=50, b=20),
    )
    return fig


def create_validation_comparison_chart() -> go.Figure:
    """
    Creates a multi-series bar and line chart comparing Actual vs Predicted values
    for Demand, Supply, and Gap across January, February, and March 2026.
    """
    months = ["Jan 2026", "Feb 2026", "Mar 2026"]
    d_act = [config.VALIDATION_BENCHMARKS[m]["demand_actual"] for m in months]
    d_pred = [config.VALIDATION_BENCHMARKS[m]["demand_predicted"] for m in months]
    s_act = [config.VALIDATION_BENCHMARKS[m]["supply_actual"] for m in months]
    s_pred = [config.VALIDATION_BENCHMARKS[m]["supply_predicted"] for m in months]

    fig = go.Figure()

    # Demand
    fig.add_trace(
        go.Bar(
            name="Actual Demand",
            x=months,
            y=d_act,
            marker_color="#2563EB",
            text=[f"{v:,.0f}" for v in d_act],
            textposition="auto",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Predicted Demand",
            x=months,
            y=d_pred,
            marker_color="#60A5FA",
            text=[f"{v:,.0f}" for v in d_pred],
            textposition="auto",
        )
    )

    # Supply
    fig.add_trace(
        go.Bar(
            name="Actual Supply",
            x=months,
            y=s_act,
            marker_color="#059669",
            text=[f"{v:,.0f}" for v in s_act],
            textposition="auto",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Predicted Supply",
            x=months,
            y=s_pred,
            marker_color="#34D399",
            text=[f"{v:,.0f}" for v in s_pred],
            textposition="auto",
        )
    )

    fig.update_layout(
        title=dict(
            text="<b>Validation Benchmarks: Actual vs Predicted (Jan - Mar 2026)</b>",
            font=dict(size=18),
        ),
        barmode="group",
        yaxis=dict(title="Electricity (MU)", gridcolor=GRID_COLOR),
        xaxis=dict(gridcolor=GRID_COLOR),
        plot_bgcolor=BG_COLOR,
        paper_bgcolor=BG_COLOR,
        height=400,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig
