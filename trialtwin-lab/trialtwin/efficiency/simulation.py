"""PROCOVA-style efficiency simulation.

Compares standard log-rank analysis vs covariate-adjusted analysis
using the digital twin's prognostic score. The key insight: adjusting
for a strong prognostic covariate reduces residual variance, yielding
narrower confidence intervals and requiring fewer patients for the
same statistical power.
"""

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter


def run_efficiency_simulation(
    os_data: pd.DataFrame,
    treatment_arms: pd.Series,
    prognostic_scores: pd.Series,
) -> dict:
    """Compare standard vs PROCOVA-style adjusted analysis.

    Returns dict with:
        standard_hr, standard_ci, adjusted_hr, adjusted_ci,
        ci_width_reduction_pct
    """
    df = pd.DataFrame({
        "os_time": os_data["os_time"].values,
        "os_event": os_data["os_event"].values,
        "treatment": (treatment_arms.values == "treatment").astype(int),
        "prognostic_score": prognostic_scores.values,
    })

    # Standard analysis: treatment only
    cph_standard = CoxPHFitter(penalizer=0.01)
    cph_standard.fit(
        df[["os_time", "os_event", "treatment"]],
        duration_col="os_time",
        event_col="os_event",
    )
    std_summary = cph_standard.summary
    standard_hr = np.exp(std_summary.loc["treatment", "coef"])
    standard_ci = (
        np.exp(std_summary.loc["treatment", "coef lower 95%"]),
        np.exp(std_summary.loc["treatment", "coef upper 95%"]),
    )

    # Adjusted analysis: treatment + prognostic score (PROCOVA-style)
    cph_adjusted = CoxPHFitter(penalizer=0.01)
    cph_adjusted.fit(
        df[["os_time", "os_event", "treatment", "prognostic_score"]],
        duration_col="os_time",
        event_col="os_event",
    )
    adj_summary = cph_adjusted.summary
    adjusted_hr = np.exp(adj_summary.loc["treatment", "coef"])
    adjusted_ci = (
        np.exp(adj_summary.loc["treatment", "coef lower 95%"]),
        np.exp(adj_summary.loc["treatment", "coef upper 95%"]),
    )

    # CI width comparison
    standard_width = standard_ci[1] - standard_ci[0]
    adjusted_width = adjusted_ci[1] - adjusted_ci[0]
    ci_width_reduction_pct = (standard_width - adjusted_width) / standard_width * 100

    return {
        "standard_hr": standard_hr,
        "standard_ci": standard_ci,
        "adjusted_hr": adjusted_hr,
        "adjusted_ci": adjusted_ci,
        "ci_width_reduction_pct": ci_width_reduction_pct,
    }
