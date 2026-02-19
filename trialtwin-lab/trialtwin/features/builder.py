"""Feature matrix assembly with missing value handling."""

import numpy as np
import pandas as pd

from trialtwin.features.baseline import extract_baseline_features
from trialtwin.features.longitudinal import extract_longitudinal_features
from trialtwin.harmonize.harmonizer import HarmonizedDataset
from trialtwin.utils.config import TrialConfig


class FeatureBuilder:
    """Orchestrates feature extraction and produces a clean feature matrix."""

    def __init__(self, config: TrialConfig):
        self.config = config
        self.baseline_cols: list[str] = []

    def build(
        self, harmonized: HarmonizedDataset, os_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Build complete feature matrix. Returns one row per subject.

        Output includes subject_id and treatment_arm for downstream filtering,
        plus all numeric feature columns with no NaN values.

        Sets self.baseline_cols to the list of baseline-only feature columns
        (excludes longitudinal features, subject_id, treatment_arm).
        """
        baseline = extract_baseline_features(harmonized.subjects)
        self.baseline_cols = [
            c for c in baseline.columns if c != "subject_id"
        ]

        if self.config.has_longitudinal and not harmonized.longitudinal.empty:
            landmark = self.config.landmark_day or 60
            longitudinal = extract_longitudinal_features(
                harmonized.longitudinal, landmark
            )
            features = baseline.merge(longitudinal, on="subject_id", how="left")
        else:
            features = baseline

        # Attach treatment arm from os_data
        features = features.merge(
            os_data[["subject_id", "treatment_arm"]], on="subject_id", how="left"
        )

        # Handle missing values: median imputation
        id_cols = ["subject_id", "treatment_arm"]
        numeric_cols = [
            c for c in features.select_dtypes(include=[np.number]).columns
            if c not in id_cols
        ]
        for col in numeric_cols:
            if features[col].isna().any():
                median_val = features[col].median()
                features[col] = features[col].fillna(median_val)

        return features
