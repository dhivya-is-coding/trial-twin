"""Longitudinal feature extraction from first N days of follow-up.

CRITICAL: Only uses data where days_since_randomization <= landmark_day
to prevent future data leakage.
"""

import numpy as np
import pandas as pd
from scipy import stats


def extract_longitudinal_features(
    longitudinal: pd.DataFrame, landmark_day: int = 60
) -> pd.DataFrame:
    """Extract features from the first landmark_day days of follow-up.

    Returns one row per subject with longitudinal-derived features.
    """
    # Strict temporal filtering — no data beyond landmark
    early = longitudinal[longitudinal["days_since_randomization"] <= landmark_day].copy()

    results = []
    for sid, group in early.groupby("subject_id"):
        row = {"subject_id": sid}
        group = group.sort_values("days_since_randomization")

        # Tumor dynamics
        if "tumor_size" in group.columns and group["tumor_size"].notna().sum() >= 2:
            tumors = group[group["tumor_size"].notna()]
            days = tumors["days_since_randomization"].values
            sizes = tumors["tumor_size"].values

            # Tumor slope (linear regression)
            if len(days) >= 2 and days[-1] > days[0]:
                slope, _, _, _, _ = stats.linregress(days, sizes)
                row["tumor_slope"] = slope
            else:
                row["tumor_slope"] = 0.0

            # Percent change from baseline
            baseline_tumor = sizes[0]
            if baseline_tumor > 0:
                row["tumor_pct_change"] = (sizes[-1] - baseline_tumor) / baseline_tumor
                row["best_response"] = (sizes.min() - baseline_tumor) / baseline_tumor
            else:
                row["tumor_pct_change"] = 0.0
                row["best_response"] = 0.0
        else:
            row["tumor_slope"] = np.nan
            row["tumor_pct_change"] = np.nan
            row["best_response"] = np.nan

        # Lab trajectory features (mean and change-from-baseline in window)
        for lab in ["hemoglobin", "albumin", "ldh", "neutrophils"]:
            if lab in group.columns and group[lab].notna().sum() >= 1:
                vals = group[lab].dropna()
                row[f"mean_{lab}_60d"] = vals.mean()
                if len(vals) >= 2:
                    row[f"delta_{lab}_60d"] = vals.iloc[-1] - vals.iloc[0]
                else:
                    row[f"delta_{lab}_60d"] = 0.0
            else:
                row[f"mean_{lab}_60d"] = np.nan
                row[f"delta_{lab}_60d"] = np.nan

        # Early adverse event count
        if "adverse_event_flag" in group.columns:
            row["early_ae_count"] = group["adverse_event_flag"].sum()
        else:
            row["early_ae_count"] = 0

        results.append(row)

    return pd.DataFrame(results)
