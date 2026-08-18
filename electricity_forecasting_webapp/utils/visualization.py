"""
Plotly Visualizations & Interactive Dashboard Charts for Electricity Forecasting.
Provides high-contrast, professional engineering charts for:
  - Forecast Comparison (Demand vs Supply with 95% Prediction Interval Error Bars)
  - Probabilistic Prediction Intervals (90%, 95%, 99% Confidence Bands with Active Focus)
  - Demand–Supply Gap Probabilistic Distribution (PDF Curve & Equilibrium Reference)
  - Risk Classification Gauge Meter (Deterministic Threshold Zones)
  - Out-of-Sample Validation Benchmarks (Actual vs Predicted Jan–Mar 2026)
  - Multi-Month Trajectory & Uncertainty Envelope
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Ensure webapp directory is in sys.path
WEBAPP_DIR = Path(__file__).resolve().parent.parent
if str(WEBAPP_DIR) not in sys.path:
    sys.path.insert(0, str(WEBAPP_DIR))

import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
import config


# Refined Engineering Dark Theme Color Palette
COLOR_DEMAND = "#3B82F6"          # Electric Blue
COLOR_DEMAND_LIGHT = "#93C5FD"    # Soft Light Blue
COLOR_SUPPLY = "#10B981"          # Emerald Green
COLOR_SUPPLY_LIGHT = "#6EE7B7"    # Soft Light Green
COLOR_GAP = "#8B5CF6"             # Royal Violet
COLOR_SHORTAGE = "#EF4444"        # Crimson Red
COLOR_SURPLUS = "#059669"         # Forest Green
COLOR_POINT = "#F59E0B"           # Amber Diamond Marker

BG_CARD = "rgba(17, 24, 39, 0.75)"       # Slate 900 Glass
BG_PAPER = "rgba(11, 15, 25, 0.0)"       # Transparent
GRID_COLOR = "rgba(148, 163, 184, 0.12)" # Soft slate grid
TEXT_COLOR = "#F8FAFC"                   # Bright White
TEXT_MUTED = "#94A3B8"                   # Muted Slate
FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"


def create_forecast_comparison_chart(
    predicted_demand: float,
    predicted_supply: float,
    predicted_gap: float,
    d_intervals: Optional[Dict[str, Any]] = None,
    s_intervals: Optional[Dict[str, Any]] = None,
    gap_intervals: Optional[Dict[str, Any]] = None,
    month_label: str = "Forecast Period",
    selected_ci: str = "95%",
) -> go.Figure:
    """
    Creates an interactive high-contrast bar chart comparing Predicted Demand,
    Predicted Supply, and Demand–Supply Gap with dynamic prediction interval error bars.
    """
    categories = ["Predicted Demand", "Predicted Supply", "Demand–Supply Gap"]
    values = [predicted_demand, predicted_supply, predicted_gap]
    colors = [
        COLOR_DEMAND,
        COLOR_SUPPLY,
        COLOR_SHORTAGE if predicted_gap > 0 else COLOR_SURPLUS,
    ]

    ci_key = "pi_95" if selected_ci == "95%" else ("pi_90" if selected_ci == "90%" else "pi_99")
    
    error_y_vals = [0.0, 0.0, 0.0]
    if d_intervals and ci_key in d_intervals:
        error_y_vals[0] = d_intervals[ci_key]["margin"]
    if s_intervals and ci_key in s_intervals:
        error_y_vals[1] = s_intervals[ci_key]["margin"]
    if gap_intervals and ci_key in gap_intervals:
        error_y_vals[2] = gap_intervals[ci_key]["margin"]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=categories,
            y=values,
            text=[f"<b>{v:,.2f}</b> MU" for v in values],
            textposition="auto",
            marker=dict(
                color=colors,
                line=dict(color="rgba(255, 255, 255, 0.25)", width=1.5),
            ),
            error_y=dict(
                type="data",
                array=error_y_vals,
                visible=True,
                color="#F8FAFC",
                thickness=2,
                width=8,
            ),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Point Forecast: <b>%{y:,.2f} MU</b><br>"
                f"{selected_ci} Uncertainty Margin: ±%{{error_y.array:,.2f}} MU<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text=f"<b>Electricity Forecast Comparison & {selected_ci} Uncertainty Margin</b> ({month_label})",
            font=dict(size=15, color=TEXT_COLOR, family=FONT_FAMILY),
        ),
        yaxis=dict(
            title=dict(text="Electricity Quantity (MU)", font=dict(color=TEXT_MUTED, size=12)),
            gridcolor=GRID_COLOR,
            zeroline=True,
            zerolinecolor="rgba(200, 200, 200, 0.35)",
            tickfont=dict(color=TEXT_MUTED),
        ),
        xaxis=dict(
            gridcolor=GRID_COLOR,
            tickfont=dict(color=TEXT_COLOR, size=12, family=FONT_FAMILY),
        ),
        plot_bgcolor=BG_PAPER,
        paper_bgcolor=BG_PAPER,
        margin=dict(l=40, r=40, t=55, b=40),
        height=380,
    )
    return fig


def create_prediction_interval_chart(
    metric_name: str,
    interval_data: Dict[str, Any],
    theme_color: str = COLOR_DEMAND,
    selected_ci: str = "95%",
) -> go.Figure:
    """
    Creates an interactive horizontal prediction interval chart with 90%, 95%, 99%
    confidence intervals, highlighting the active confidence level (95% default).
    """
    point = interval_data["point_forecast"]
    sigma = interval_data["sigma"]
    pi_90 = interval_data["pi_90"]
    pi_95 = interval_data["pi_95"]
    pi_99 = interval_data["pi_99"]

    fig = go.Figure()

    tiers = [
        ("99% Interval", pi_99, "#A855F7", 9 if selected_ci == "99%" else 4, 1.0 if selected_ci == "99%" else 0.45),
        ("95% Interval", pi_95, theme_color, 11 if selected_ci == "95%" else 5, 1.0 if selected_ci == "95%" else 0.5),
        ("90% Interval", pi_90, "#10B981", 9 if selected_ci == "90%" else 4, 1.0 if selected_ci == "90%" else 0.45),
    ]

    for label, data, color, width, opacity in tiers:
        is_active = (selected_ci in label)
        fig.add_trace(
            go.Scatter(
                x=[data["lower"], data["upper"]],
                y=[label, label],
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=width),
                opacity=opacity,
                marker=dict(size=14 if is_active else 8, symbol="line-ns-open", line=dict(width=3, color=color)),
                hovertemplate=(
                    f"<b>{label} ({metric_name})</b><br>"
                    f"Lower Bound: <b>{data['lower']:,.2f} MU</b><br>"
                    f"Upper Bound: <b>{data['upper']:,.2f} MU</b><br>"
                    f"Span: {data['upper'] - data['lower']:,.2f} MU (±{data['margin']:,.2f} MU)<extra></extra>"
                ),
            )
        )

    for label, _, _, _, _ in tiers:
        fig.add_trace(
            go.Scatter(
                x=[point],
                y=[label],
                mode="markers",
                name="Point Forecast",
                showlegend=(label == "99% Interval"),
                marker=dict(color=COLOR_POINT, size=13, symbol="diamond", line=dict(width=1.5, color="#FFFFFF")),
                hovertemplate=f"<b>Point Forecast</b>: {point:,.2f} MU<extra></extra>",
            )
        )

    fig.add_vline(
        x=point,
        line_width=1.5,
        line_dash="dash",
        line_color=COLOR_POINT,
        annotation_text=f"Point: {point:,.2f} MU",
        annotation_position="top right",
        annotation_font=dict(color=COLOR_POINT, size=11, family=FONT_FAMILY),
    )

    fig.update_layout(
        title=dict(
            text=f"<b>{metric_name} — Probabilistic Prediction Intervals</b> (σ = {sigma:.2f} MU)",
            font=dict(size=14, color=TEXT_COLOR, family=FONT_FAMILY),
        ),
        xaxis=dict(
            title=dict(text="Electricity (MU)", font=dict(color=TEXT_MUTED, size=11)),
            gridcolor=GRID_COLOR,
            tickfont=dict(color=TEXT_MUTED),
        ),
        yaxis=dict(
            gridcolor=GRID_COLOR,
            tickfont=dict(color=TEXT_COLOR, size=11, family=FONT_FAMILY),
        ),
        plot_bgcolor=BG_PAPER,
        paper_bgcolor=BG_PAPER,
        margin=dict(l=30, r=30, t=50, b=35),
        height=310,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=TEXT_MUTED, size=10),
        ),
    )
    return fig


def create_gap_distribution_chart(
    predicted_gap: float,
    gap_sigma: float,
    gap_intervals: Dict[str, Any],
    selected_ci: str = "95%",
) -> go.Figure:
    """
    Creates the normal probability density function (PDF) curve for the forecasted gap
    highlighting the 0 MU balance line, risk regions, and the active confidence interval.
    """
    x_min = predicted_gap - 3.8 * gap_sigma
    x_max = predicted_gap + 3.8 * gap_sigma
    x = np.linspace(x_min, x_max, 400)
    pdf = norm.pdf(x, loc=predicted_gap, scale=gap_sigma)

    ci_key = "pi_95" if selected_ci == "95%" else ("pi_90" if selected_ci == "90%" else "pi_99")
    active_pi = gap_intervals[ci_key]
    
    x_ci = np.linspace(active_pi["lower"], active_pi["upper"], 200)
    pdf_ci = norm.pdf(x_ci, loc=predicted_gap, scale=gap_sigma)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=pdf,
            mode="lines",
            name="Gap Probability Density",
            line=dict(color="#A78BFA", width=2.5),
            hovertemplate="Gap: <b>%{x:,.2f} MU</b><br>Density: %{y:.6f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=np.concatenate([x_ci, x_ci[::-1]]),
            y=np.concatenate([pdf_ci, np.zeros_like(pdf_ci)]),
            fill="toself",
            fillcolor="rgba(139, 92, 246, 0.3)",
            line=dict(color="rgba(255,255,255,0)"),
            name=f"{selected_ci} Confidence Interval",
            hoverinfo="skip",
        )
    )

    fig.add_vline(
        x=predicted_gap,
        line_width=2,
        line_color=COLOR_POINT,
        annotation_text=f"Expected Gap: {predicted_gap:,.2f} MU",
        annotation_position="top left",
        annotation_font=dict(color=COLOR_POINT, size=11, family=FONT_FAMILY),
    )

    fig.add_vline(
        x=0,
        line_width=1.5,
        line_dash="dot",
        line_color="#10B981",
        annotation_text="Equilibrium (0 MU)",
        annotation_position="bottom right",
        annotation_font=dict(color="#10B981", size=10, family=FONT_FAMILY),
    )

    fig.add_vline(
        x=3000.0,
        line_width=1,
        line_dash="dash",
        line_color="#F59E0B",
        annotation_text="Mod Threshold (3000 MU)",
        annotation_position="bottom left",
        annotation_font=dict(color="#F59E0B", size=9, family=FONT_FAMILY),
    )

    fig.add_vline(
        x=4500.0,
        line_width=1,
        line_dash="dash",
        line_color="#EF4444",
        annotation_text="High Threshold (4500 MU)",
        annotation_position="top right",
        annotation_font=dict(color="#EF4444", size=9, family=FONT_FAMILY),
    )

    fig.update_layout(
        title=dict(
            text=f"<b>Probabilistic Distribution of Demand–Supply Gap</b> (σ_Gap = {gap_sigma:,.2f} MU)",
            font=dict(size=14, color=TEXT_COLOR, family=FONT_FAMILY),
        ),
        xaxis=dict(
            title=dict(text="Forecasted Gap in MU  [+ Shortage / Deficit | - Surplus]", font=dict(color=TEXT_MUTED, size=11)),
            gridcolor=GRID_COLOR,
            tickfont=dict(color=TEXT_MUTED),
        ),
        yaxis=dict(
            title=dict(text="Probability Density", font=dict(color=TEXT_MUTED, size=11)),
            gridcolor=GRID_COLOR,
            showticklabels=False,
        ),
        plot_bgcolor=BG_PAPER,
        paper_bgcolor=BG_PAPER,
        margin=dict(l=30, r=30, t=50, b=35),
        height=340,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=TEXT_MUTED, size=10),
        ),
    )
    return fig


def create_risk_gauge(predicted_gap: float, risk_level: str) -> go.Figure:
    """
    Creates an interactive gauge meter showing the position of the forecasted gap
    relative to project-defined risk thresholds (0 - 3000 Low, 3000 - 4500 Moderate, >4500 High).
    """
    max_gauge = max(6000.0, predicted_gap * 1.25)
    gauge_val = max(0.0, predicted_gap)

    bar_color = "#3B82F6" if risk_level == "Low" else ("#F59E0B" if risk_level == "Moderate" else "#EF4444")

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=gauge_val,
            domain={"x": [0, 1], "y": [0, 1]},
            title={
                "text": f"<b>Risk Classification Gauge</b><br><span style='font-size:12px;color:#94A3B8'>Severity: {risk_level.upper()} RISK</span>",
                "font": {"size": 15, "color": TEXT_COLOR, "family": FONT_FAMILY},
            },
            number={"suffix": " MU", "font": {"size": 22, "color": TEXT_COLOR, "family": FONT_FAMILY}},
            delta={"reference": 3000, "increasing": {"color": "#EF4444"}, "decreasing": {"color": "#10B981"}},
            gauge={
                "axis": {"range": [0, max_gauge], "tickwidth": 1, "tickcolor": TEXT_MUTED, "tickfont": {"color": TEXT_MUTED, "size": 10}},
                "bar": {"color": bar_color, "thickness": 0.3},
                "bgcolor": "rgba(30, 41, 59, 0.5)",
                "borderwidth": 1,
                "bordercolor": "rgba(255, 255, 255, 0.1)",
                "steps": [
                    {"range": [0, 3000], "color": "rgba(59, 130, 246, 0.2)"},
                    {"range": [3000, 4500], "color": "rgba(245, 158, 11, 0.25)"},
                    {"range": [4500, max_gauge], "color": "rgba(239, 68, 68, 0.3)"},
                ],
                "threshold": {
                    "line": {"color": "#EF4444", "width": 3},
                    "thickness": 0.8,
                    "value": 4500,
                },
            },
        )
    )

    fig.update_layout(
        plot_bgcolor=BG_PAPER,
        paper_bgcolor=BG_PAPER,
        height=280,
        margin=dict(l=25, r=25, t=45, b=15),
    )
    return fig


def create_validation_comparison_chart() -> go.Figure:
    """
    Creates a multi-series grouped bar chart comparing Actual vs Predicted values
    for Demand and Supply across January, February, and March 2026 hold-out benchmarks.
    """
    months = ["Jan 2026", "Feb 2026", "Mar 2026"]
    d_act = [config.VALIDATION_BENCHMARKS[m]["demand_actual"] for m in months]
    d_pred = [config.VALIDATION_BENCHMARKS[m]["demand_predicted"] for m in months]
    s_act = [config.VALIDATION_BENCHMARKS[m]["supply_actual"] for m in months]
    s_pred = [config.VALIDATION_BENCHMARKS[m]["supply_predicted"] for m in months]

    fig = go.Figure()

    # Demand Pairs
    fig.add_trace(
        go.Bar(
            name="Actual Demand",
            x=months,
            y=d_act,
            marker_color="#2563EB",
            text=[f"{v:,.0f}" for v in d_act],
            textposition="auto",
            hovertemplate="Actual Demand: <b>%{y:,.2f} MU</b><extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Predicted Demand",
            x=months,
            y=d_pred,
            marker_color="#93C5FD",
            text=[f"{v:,.0f}" for v in d_pred],
            textposition="auto",
            hovertemplate="Predicted Demand: <b>%{y:,.2f} MU</b><extra></extra>",
        )
    )

    # Supply Pairs
    fig.add_trace(
        go.Bar(
            name="Actual Supply",
            x=months,
            y=s_act,
            marker_color="#059669",
            text=[f"{v:,.0f}" for v in s_act],
            textposition="auto",
            hovertemplate="Actual Supply: <b>%{y:,.2f} MU</b><extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Predicted Supply",
            x=months,
            y=s_pred,
            marker_color="#6EE7B7",
            text=[f"{v:,.0f}" for v in s_pred],
            textposition="auto",
            hovertemplate="Predicted Supply: <b>%{y:,.2f} MU</b><extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text="<b>Hold-Out Validation Benchmarks: Actual vs Predicted (Jan – Mar 2026)</b>",
            font=dict(size=15, color=TEXT_COLOR, family=FONT_FAMILY),
        ),
        barmode="group",
        yaxis=dict(
            title=dict(text="Electricity (MU)", font=dict(color=TEXT_MUTED, size=11)),
            gridcolor=GRID_COLOR,
            tickfont=dict(color=TEXT_MUTED),
        ),
        xaxis=dict(
            gridcolor=GRID_COLOR,
            tickfont=dict(color=TEXT_COLOR, size=11, family=FONT_FAMILY),
        ),
        plot_bgcolor=BG_PAPER,
        paper_bgcolor=BG_PAPER,
        height=380,
        margin=dict(l=35, r=35, t=55, b=35),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=TEXT_MUTED, size=10),
        ),
    )
    return fig


def create_multi_month_timeline_chart() -> go.Figure:
    """
    Creates a trajectory chart displaying Demand and Supply over Jan–Mar 2026
    along with shaded 95% prediction interval ribbons.
    """
    months = ["Jan 2026", "Feb 2026", "Mar 2026"]
    
    d_pred = [config.VALIDATION_BENCHMARKS[m]["demand_predicted"] for m in months]
    d_act = [config.VALIDATION_BENCHMARKS[m]["demand_actual"] for m in months]
    
    s_pred = [config.VALIDATION_BENCHMARKS[m]["supply_predicted"] for m in months]
    s_act = [config.VALIDATION_BENCHMARKS[m]["supply_actual"] for m in months]
    
    # 95% bounds
    z = config.Z_95
    d_upper = [p + z * config.BASE_SIGMA_DEMAND for p in d_pred]
    d_lower = [p - z * config.BASE_SIGMA_DEMAND for p in d_pred]
    
    s_upper = [p + z * config.BASE_SIGMA_SUPPLY for p in s_pred]
    s_lower = [p - z * config.BASE_SIGMA_SUPPLY for p in s_pred]

    fig = go.Figure()

    # Demand Ribbon
    fig.add_trace(
        go.Scatter(
            x=months + months[::-1],
            y=d_upper + d_lower[::-1],
            fill="toself",
            fillcolor="rgba(59, 130, 246, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Demand 95% PI",
            hoverinfo="skip",
        )
    )

    # Demand Prediction Line
    fig.add_trace(
        go.Scatter(
            x=months,
            y=d_pred,
            mode="lines+markers",
            name="Predicted Demand",
            line=dict(color=COLOR_DEMAND, width=3),
            marker=dict(size=8, symbol="circle"),
            hovertemplate="Pred Demand: <b>%{y:,.2f} MU</b><extra></extra>",
        )
    )

    # Demand Actual Line
    fig.add_trace(
        go.Scatter(
            x=months,
            y=d_act,
            mode="markers",
            name="Actual Demand",
            marker=dict(color="#60A5FA", size=10, symbol="x"),
            hovertemplate="Actual Demand: <b>%{y:,.2f} MU</b><extra></extra>",
        )
    )

    # Supply Ribbon
    fig.add_trace(
        go.Scatter(
            x=months + months[::-1],
            y=s_upper + s_lower[::-1],
            fill="toself",
            fillcolor="rgba(16, 185, 129, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Supply 95% PI",
            hoverinfo="skip",
        )
    )

    # Supply Prediction Line
    fig.add_trace(
        go.Scatter(
            x=months,
            y=s_pred,
            mode="lines+markers",
            name="Predicted Supply",
            line=dict(color=COLOR_SUPPLY, width=3),
            marker=dict(size=8, symbol="circle"),
            hovertemplate="Pred Supply: <b>%{y:,.2f} MU</b><extra></extra>",
        )
    )

    # Supply Actual Line
    fig.add_trace(
        go.Scatter(
            x=months,
            y=s_act,
            mode="markers",
            name="Actual Supply",
            marker=dict(color="#34D399", size=10, symbol="x"),
            hovertemplate="Actual Supply: <b>%{y:,.2f} MU</b><extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text="<b>Forward Operational Trajectory & 95% Prediction Envelope (Jan – Mar 2026)</b>",
            font=dict(size=15, color=TEXT_COLOR, family=FONT_FAMILY),
        ),
        yaxis=dict(
            title=dict(text="Electricity (MU)", font=dict(color=TEXT_MUTED, size=11)),
            gridcolor=GRID_COLOR,
            tickfont=dict(color=TEXT_MUTED),
        ),
        xaxis=dict(
            gridcolor=GRID_COLOR,
            tickfont=dict(color=TEXT_COLOR, size=11, family=FONT_FAMILY),
        ),
        plot_bgcolor=BG_PAPER,
        paper_bgcolor=BG_PAPER,
        height=360,
        margin=dict(l=35, r=35, t=55, b=35),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=TEXT_MUTED, size=10),
        ),
    )
    return fig
