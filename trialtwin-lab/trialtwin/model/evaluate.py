"""Model evaluation: C-index, calibration, and discrimination plots."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index


def evaluate_model(model, X: pd.DataFrame, os_data: pd.DataFrame) -> dict:
    """Evaluate a survival model on held-out data.

    Works with both CoxPHModel and GBMSurvivalModel (duck typing on predict_risk_score).
    """
    risk_scores = model.predict_risk_score(X)
    c_index = concordance_index(
        os_data["os_time"].values,
        -risk_scores.values,  # negate because higher risk = shorter survival
        os_data["os_event"].values,
    )
    return {"c_index": c_index}


def save_evaluation_plots(
    model, X: pd.DataFrame, os_data: pd.DataFrame, fig_dir: Path
) -> None:
    """Generate and save evaluation plots."""
    fig_dir.mkdir(parents=True, exist_ok=True)

    risk_scores = model.predict_risk_score(X)

    # KM curves by risk tercile (discrimination)
    terciles = pd.qcut(risk_scores, 3, labels=["Low Risk", "Medium Risk", "High Risk"])
    fig, ax = plt.subplots(figsize=(8, 6))
    kmf = KaplanMeierFitter()

    for label in ["Low Risk", "Medium Risk", "High Risk"]:
        mask = terciles == label
        kmf.fit(
            os_data["os_time"][mask],
            event_observed=os_data["os_event"][mask],
            label=label,
        )
        kmf.plot_survival_function(ax=ax)

    ax.set_xlabel("Days")
    ax.set_ylabel("Survival Probability")
    ax.set_title("Kaplan-Meier Curves by Predicted Risk Group")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "km_by_risk_group.png", dpi=150)
    plt.close(fig)
