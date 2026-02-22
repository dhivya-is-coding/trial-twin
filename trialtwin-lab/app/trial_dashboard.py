"""Trial Dashboard tab — aggregate results and efficiency analysis."""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from lifelines import KaplanMeierFitter
import streamlit as st


def render_trial_dashboard(artifacts: dict, report_dir: Path):
    """Render the trial-level dashboard with KM curves and efficiency results."""
    os_data = artifacts["os_data"]
    efficiency = artifacts["efficiency"]
    metrics = artifacts["metrics"]
    twins = artifacts.get("twins")
    effects = artifacts.get("effects")

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("N Subjects", os_data.shape[0])
    with col2:
        st.metric("OS Events", f"{os_data['os_event'].sum():.0f}")
    with col3:
        st.metric("Cox C-index", f"{metrics['cox_c_index']:.3f}")
    with col4:
        st.metric(
            "Treatment HR",
            f"{efficiency['adjusted_hr']:.3f}",
            delta=f"{efficiency['ci_width_reduction_pct']:.1f}% CI reduction",
        )

    st.divider()

    # KM curves by arm (interactive plotly)
    st.subheader("Kaplan-Meier Survival Curves by Treatment Arm")
    fig_km = _plot_km_plotly(os_data)
    st.plotly_chart(fig_km, width='stretch')

    # Efficiency comparison
    st.subheader("PROCOVA-style Efficiency Gain")
    col_left, col_right = st.columns(2)

    with col_left:
        fig_eff = _plot_efficiency_plotly(efficiency)
        st.plotly_chart(fig_eff, width='stretch')

    with col_right:
        st.markdown("**Standard Analysis (treatment only)**")
        st.write(f"HR: {efficiency['standard_hr']:.3f} "
                 f"(95% CI: {efficiency['standard_ci'][0]:.3f}–{efficiency['standard_ci'][1]:.3f})")

        st.markdown("**Adjusted Analysis (treatment + prognostic score)**")
        st.write(f"HR: {efficiency['adjusted_hr']:.3f} "
                 f"(95% CI: {efficiency['adjusted_ci'][0]:.3f}–{efficiency['adjusted_ci'][1]:.3f})")

        st.markdown(f"**CI Width Reduction: {efficiency['ci_width_reduction_pct']:.1f}%**")
        st.caption(
            "By adjusting for the digital twin's prognostic score, "
            "we get a more precise treatment effect estimate — requiring "
            "fewer patients for the same statistical power."
        )

    # Digital twin summary
    if twins is not None and effects is not None:
        st.divider()
        st.subheader("Digital Twin Summary")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Treated Patients", len(twins))
        with col_b:
            st.metric("Mean Predicted Control Median", f"{twins['predicted_median_survival'].mean():.0f} days")
        with col_c:
            te_mean = effects["treatment_effect"].mean()
            st.metric("Mean Treatment Effect", f"{te_mean:+.0f} days")

        # Distribution of treatment effects
        fig_te = go.Figure(go.Histogram(
            x=effects["treatment_effect"],
            nbinsx=30,
            marker_color="#45b39d",
            opacity=0.8,
        ))
        fig_te.add_vline(x=0, line_dash="dash", line_color="red",
                         annotation_text="No effect")
        fig_te.update_layout(
            xaxis_title="Individual Treatment Effect (days)",
            yaxis_title="Count",
            title="Distribution of Individual Treatment Effects",
            height=350,
        )
        st.plotly_chart(fig_te, width='stretch')

    # QA report
    st.divider()
    st.subheader("Data Quality Report")
    qa_path = report_dir / "qa_report.md"
    if qa_path.exists():
        st.markdown(qa_path.read_text())
    else:
        st.info("QA report not found. Run `make data` to generate it.")


def _plot_km_plotly(os_data: pd.DataFrame) -> go.Figure:
    """Create interactive Plotly KM curves by treatment arm."""
    fig = go.Figure()
    kmf = KaplanMeierFitter()

    colors = {"control": "#e74c3c", "treatment": "#4a90d9"}
    for arm, color in colors.items():
        subset = os_data[os_data["treatment_arm"] == arm]
        kmf.fit(subset["os_time"], event_observed=subset["os_event"], label=arm)

        timeline = kmf.survival_function_.index.values
        survival = kmf.survival_function_.iloc[:, 0].values
        ci_lower = kmf.confidence_interval_.iloc[:, 0].values
        ci_upper = kmf.confidence_interval_.iloc[:, 1].values

        # Confidence band
        fig.add_trace(go.Scatter(
            x=np.concatenate([timeline, timeline[::-1]]),
            y=np.concatenate([ci_upper, ci_lower[::-1]]),
            fill="toself", fillcolor=color, opacity=0.15,
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))

        # Main curve
        fig.add_trace(go.Scatter(
            x=timeline, y=survival,
            mode="lines", name=arm.capitalize(),
            line=dict(color=color, width=2.5),
        ))

    fig.update_layout(
        xaxis_title="Days from Randomization",
        yaxis_title="Overall Survival Probability",
        yaxis_range=[0, 1.05],
        height=450,
        legend=dict(x=0.7, y=0.95),
    )
    return fig


def _plot_efficiency_plotly(efficiency: dict) -> go.Figure:
    """Bar chart comparing standard vs adjusted CI width."""
    std_w = efficiency["standard_ci"][1] - efficiency["standard_ci"][0]
    adj_w = efficiency["adjusted_ci"][1] - efficiency["adjusted_ci"][0]

    fig = go.Figure(go.Bar(
        x=["Standard", "PROCOVA-style"],
        y=[std_w, adj_w],
        marker_color=["#4a90d9", "#45b39d"],
        text=[f"{std_w:.3f}", f"{adj_w:.3f}"],
        textposition="auto",
    ))
    fig.update_layout(
        yaxis_title="95% CI Width (HR)",
        title=f"{efficiency['ci_width_reduction_pct']:.1f}% Narrower Confidence Interval",
        height=350,
    )
    return fig
