"""Digital Twin Generator.

Generates counterfactual control-arm predictions for treated patients.
The prognostic score computed here is the key input to PROCOVA-style
covariate adjustment — directly mirroring Unlearn's methodology.
"""

import numpy as np
import pandas as pd


class DigitalTwinGenerator:
    """Generate digital twins using a fitted control-arm survival model.

    A digital twin predicts what a treated patient's outcome would have been
    under the control condition. The scalar prognostic score summarizes this
    prediction and is used as a covariate in efficiency analysis.
    """

    def __init__(self, model):
        """Initialize with a fitted control-arm model (CoxPHModel or GBMSurvivalModel)."""
        self.model = model

    def generate_twins(
        self, X: pd.DataFrame, subject_ids: pd.Series
    ) -> pd.DataFrame:
        """Generate digital twin predictions for each subject.

        Returns DataFrame with:
            subject_id, prognostic_score, predicted_median_survival,
            predicted_6mo_surv, predicted_12mo_surv, predicted_18mo_surv
        """
        risk_scores = self.model.predict_risk_score(X)
        sf = self.model.predict_survival_function(X)

        # Extract survival probabilities at specific timepoints
        timepoints = {"6mo": 183, "12mo": 365, "18mo": 548}
        surv_at = {}
        for name, days in timepoints.items():
            # Find nearest timepoint in survival function index
            idx = sf.index
            nearest = idx[np.argmin(np.abs(idx - days))]
            surv_at[name] = sf.loc[nearest].values

        # Median survival: time when S(t) crosses 0.5
        medians = []
        for col in sf.columns:
            curve = sf[col]
            below_half = curve[curve <= 0.5]
            if not below_half.empty:
                medians.append(below_half.index[0])
            else:
                medians.append(curve.index[-1])  # censored: use last timepoint

        return pd.DataFrame({
            "subject_id": subject_ids.values,
            "prognostic_score": risk_scores.values,
            "predicted_median_survival": medians,
            "predicted_6mo_surv": surv_at["6mo"],
            "predicted_12mo_surv": surv_at["12mo"],
            "predicted_18mo_surv": surv_at["18mo"],
        })

    def compute_prognostic_score(self, X: pd.DataFrame) -> pd.Series:
        """Compute scalar prognostic score per patient.

        This is the key output used in PROCOVA-style analysis:
        a single number summarizing predicted control outcome.
        """
        return self.model.predict_risk_score(X)

    def estimate_individual_treatment_effect(
        self, twins: pd.DataFrame, os_treated: pd.DataFrame
    ) -> pd.DataFrame:
        """Compare observed treated outcome to digital twin prediction.

        treatment_effect = observed_os_time - predicted_median_survival
        Positive values indicate the treatment helped (patient survived
        longer than predicted under control).
        """
        merged = twins[["subject_id", "predicted_median_survival"]].merge(
            os_treated[["subject_id", "os_time", "os_event"]],
            on="subject_id",
            how="inner",
        )
        merged["treatment_effect"] = merged["os_time"] - merged["predicted_median_survival"]
        return merged
