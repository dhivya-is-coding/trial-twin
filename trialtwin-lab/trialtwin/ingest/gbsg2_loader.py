"""GBSG2 real breast cancer data adapter.

Wraps the German Breast Cancer Study Group 2 dataset (686 patients)
from scikit-survival into our unified DataSource interface.
"""

import numpy as np
import pandas as pd
from sksurv.datasets import load_gbsg2

from trialtwin.ingest.base import DataSource
from trialtwin.utils.config import TrialConfig


class GBSG2Source(DataSource):
    """Load real GBSG2 breast cancer data and map to unified schema."""

    def __init__(self, config: TrialConfig):
        self.config = config

    def load(self) -> dict[str, pd.DataFrame]:
        X, y = load_gbsg2()

        n = len(X)
        mappings = self.config.get("baseline_mappings")
        arm_map = self.config.get("column_mappings", "treatment_values")

        # Map treatment arm
        arm_col = self.config.get("column_mappings", "treatment_arm")
        arms = X[arm_col].map({arm_map["treatment"]: "treatment", arm_map["control"]: "control"})

        subjects = pd.DataFrame({
            "subject_id": [f"GBSG2-{i:04d}" for i in range(n)],
            "trial_id": self.config.trial_id,
            "treatment_arm": arms.values,
            "randomization_date": pd.Timestamp("2000-01-01"),  # not available
            "age": X[mappings["age"]].values,
            "sex": "F",  # all female (breast cancer)
            "race": "UNKNOWN",  # not available
            "ecog": np.nan,  # not available in GBSG2
            "tumor_burden": X[mappings["tumor_size"]].values,
            "tumor_grade": X[mappings["tumor_grade"]].values.astype(str),
            "positive_nodes": X[mappings["positive_nodes"]].values,
            "progesterone_receptor": X[mappings["progesterone_receptor"]].values,
            "estrogen_receptor": X[mappings["estrogen_receptor"]].values,
            "menopausal_status": X[mappings["menopausal_status"]].values.astype(str),
        })

        # Extract survival endpoints from structured array
        os_event = y["cens"].astype(int)
        os_time = y["time"].astype(float)

        endpoints = pd.DataFrame({
            "subject_id": subjects["subject_id"],
            "os_time": os_time,
            "os_event": os_event,
            "pfs_time": np.nan,  # not available
            "pfs_event": np.nan,
        })

        # No longitudinal data in GBSG2
        longitudinal = pd.DataFrame(columns=[
            "subject_id", "days_since_randomization", "visit_number",
            "tumor_size", "adverse_event_flag",
        ])

        return {
            "subjects": subjects,
            "longitudinal": longitudinal,
            "endpoints": endpoints,
        }
