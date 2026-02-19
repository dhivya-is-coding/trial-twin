"""Gradient Boosting Survival model wrapper.

Wraps scikit-survival's GradientBoostingSurvivalAnalysis with a clean
interface matching our CoxPHModel API.
"""

import numpy as np
import pandas as pd
from sksurv.ensemble import GradientBoostingSurvivalAnalysis


def _make_survival_target(event: pd.Series, time: pd.Series) -> np.ndarray:
    """Convert event/time columns to scikit-survival structured array."""
    return np.array(
        list(zip(event.astype(bool), time.astype(float))),
        dtype=[("event", bool), ("time", float)],
    )


class GBMSurvivalModel:
    """Gradient boosting survival model trained on control arm data."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
    ):
        self.model = GradientBoostingSurvivalAnalysis(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=0.8,
            random_state=42,
        )
        self._feature_cols: list[str] = []

    def fit(
        self, X: pd.DataFrame, duration: pd.Series, event: pd.Series
    ) -> "GBMSurvivalModel":
        self._feature_cols = list(X.columns)
        y = _make_survival_target(event, duration)
        self.model.fit(X.values, y)
        return self

    def predict_risk_score(self, X: pd.DataFrame) -> pd.Series:
        """Predict risk scores (higher = higher risk)."""
        return pd.Series(self.model.predict(X[self._feature_cols].values), index=X.index)

    def predict_survival_function(self, X: pd.DataFrame) -> list:
        """Predict individual survival functions (list of StepFunction)."""
        return self.model.predict_survival_function(X[self._feature_cols].values)
