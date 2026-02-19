"""Cox Proportional Hazards model wrapper.

Wraps lifelines.CoxPHFitter with a clean interface that separates
features (X) from duration and event for fitting and prediction.
"""

import pandas as pd
from lifelines import CoxPHFitter


class CoxPHModel:
    """Cox PH model trained on control arm data."""

    VARIANCE_THRESHOLD = 1e-3

    def __init__(self, penalizer: float = 0.1):
        self.penalizer = penalizer
        self.model = CoxPHFitter(penalizer=penalizer)
        self._feature_cols: list[str] = []
        self._dropped_cols: list[str] = []

    def fit(
        self, X: pd.DataFrame, duration: pd.Series, event: pd.Series
    ) -> "CoxPHModel":
        """Fit Cox PH model.

        lifelines requires a single DataFrame with duration+event columns.
        This wrapper handles the concat internally. Automatically drops
        near-zero-variance columns to prevent convergence failures.
        """
        # Drop columns with near-zero variance in the fitting data
        variances = X.var()
        self._dropped_cols = list(variances[variances < self.VARIANCE_THRESHOLD].index)
        self._feature_cols = [c for c in X.columns if c not in self._dropped_cols]

        df = X[self._feature_cols].copy()
        df["_duration"] = duration.values
        df["_event"] = event.values
        self.model.fit(df, duration_col="_duration", event_col="_event")
        return self

    def summary(self) -> pd.DataFrame:
        return self.model.summary

    def predict_survival_function(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict individual survival curves.

        Returns DataFrame: rows = timepoints, columns = subjects.
        """
        return self.model.predict_survival_function(X[self._feature_cols])

    def predict_median(self, X: pd.DataFrame) -> pd.Series:
        """Predict median survival time for each subject."""
        return self.model.predict_median(X[self._feature_cols]).squeeze()

    def predict_risk_score(self, X: pd.DataFrame) -> pd.Series:
        """Compute linear predictor (X @ beta) — the prognostic score."""
        return self.model.predict_partial_hazard(X[self._feature_cols]).squeeze()
