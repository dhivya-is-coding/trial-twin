"""Patient Explorer tab — individual digital twin analysis."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render_patient_explorer(artifacts: dict):
    """Render the patient-level digital twin explorer."""
    features = artifacts["features"]
    os_data = artifacts["os_data"]
    twins = artifacts.get("twins")
    effects = artifacts.get("effects")
    cox = artifacts["cox_model"]

    # Only show treated patients (they have digital twins)
    treated = features[features["treatment_arm"] == "treatment"].copy()
    treated_ids = treated["subject_id"].tolist()

    if not treated_ids:
        st.warning("No treated patients found.")
        return

    patient_id = st.sidebar.selectbox("Select Patient", treated_ids)
    patient_row = treated[treated["subject_id"] == patient_id].iloc[0]
    patient_os = os_data[os_data["subject_id"] == patient_id].iloc[0]

    # Demographics card
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Age", f"{patient_row.get('age', 'N/A')}")
        sex_val = "Male" if patient_row.get("sex_male", 0) == 1 else "Female"
        st.metric("Sex", sex_val)
    with col2:
        st.metric("ECOG", f"{patient_row.get('ecog', 'N/A'):.0f}")
        st.metric("Arm", "Treatment")
    with col3:
        event_str = "Deceased" if patient_os["os_event"] == 1 else "Censored"
        st.metric("OS Time", f"{patient_os['os_time']:.0f} days")
        st.metric("Status", event_str)

    st.divider()

    # Predicted survival curve
    st.subheader("Predicted Control-Arm Survival Curve (Digital Twin)")
    baseline_cols = cox._feature_cols
    patient_features = patient_row[baseline_cols].to_frame().T.astype(float)

    sf = cox.predict_survival_function(patient_features)
    times = sf.index.values
    surv_probs = sf.iloc[:, 0].values

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=surv_probs,
        mode="lines", name="Predicted S(t) under control",
        line=dict(color="#4a90d9", width=2),
    ))

    # Mark actual observed time
    fig.add_vline(
        x=patient_os["os_time"], line_dash="dash", line_color="red",
        annotation_text=f"Observed: {patient_os['os_time']:.0f}d ({event_str})",
    )

    # Mark predicted median
    if twins is not None:
        twin_row = twins[twins["subject_id"] == patient_id]
        if not twin_row.empty:
            pred_median = twin_row.iloc[0]["predicted_median_survival"]
            fig.add_vline(
                x=pred_median, line_dash="dot", line_color="green",
                annotation_text=f"Predicted median: {pred_median:.0f}d",
            )

    fig.update_layout(
        xaxis_title="Days from Randomization",
        yaxis_title="Survival Probability",
        yaxis_range=[0, 1.05],
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Treatment effect
    if effects is not None:
        effect_row = effects[effects["subject_id"] == patient_id]
        if not effect_row.empty:
            te = effect_row.iloc[0]["treatment_effect"]
            st.metric(
                "Individual Treatment Effect",
                f"{te:+.0f} days",
                help="Observed OS time minus predicted control median. "
                     "Positive = patient lived longer than predicted under control.",
            )

    # Feature contributions
    st.subheader("Top Feature Contributions to Risk Score")
    coefs = cox.model.summary["coef"]
    patient_vals = patient_features.iloc[0]
    contributions = (coefs * patient_vals).sort_values(key=abs, ascending=False).head(5)

    fig2 = go.Figure(go.Bar(
        x=contributions.values,
        y=contributions.index,
        orientation="h",
        marker_color=["#e74c3c" if v > 0 else "#2ecc71" for v in contributions.values],
    ))
    fig2.update_layout(
        xaxis_title="Contribution to Log-Hazard (coef × value)",
        height=250,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Red = increases risk, Green = decreases risk")
