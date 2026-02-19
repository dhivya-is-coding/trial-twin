"""Survival endpoint derivation.

Derives Overall Survival (OS) and Progression-Free Survival (PFS)
from harmonized clinical trial data.
"""

import numpy as np
import pandas as pd


def derive_overall_survival(
    subjects: pd.DataFrame, endpoints: pd.DataFrame
) -> pd.DataFrame:
    """Derive Overall Survival endpoint.

    Definition:
        Time: days from randomization to death (or censoring)
        Event: 1 if death observed, 0 if censored
        Censoring: last known follow-up if no death recorded

    Returns DataFrame with: subject_id, os_time, os_event, treatment_arm
    """
    os_data = endpoints[["subject_id", "os_time", "os_event"]].copy()
    os_data = os_data.merge(
        subjects[["subject_id", "treatment_arm"]], on="subject_id", how="left"
    )
    os_data["os_event"] = os_data["os_event"].astype(int)
    os_data["os_time"] = os_data["os_time"].astype(float)
    return os_data


def derive_pfs(
    subjects: pd.DataFrame,
    longitudinal: pd.DataFrame,
    endpoints: pd.DataFrame,
) -> pd.DataFrame:
    """Derive Progression-Free Survival from tumor assessments.

    Definition:
        Event: first of (tumor progression, death)
        Time: first_event_date - randomization_date (days)
        Progression: >= 20% increase in tumor sum from nadir (RECIST-simplified)
        Censoring: last tumor assessment date if no event

    Returns DataFrame with: subject_id, pfs_time, pfs_event
    """
    # If PFS already computed (synthetic data), use it directly
    if "pfs_time" in endpoints.columns and endpoints["pfs_time"].notna().all():
        return endpoints[["subject_id", "pfs_time", "pfs_event"]].copy()

    # Otherwise derive from longitudinal tumor data
    results = []
    for sid in endpoints["subject_id"].unique():
        os_time = endpoints.loc[endpoints["subject_id"] == sid, "os_time"].values[0]
        os_event = endpoints.loc[endpoints["subject_id"] == sid, "os_event"].values[0]

        subj_tumors = longitudinal[
            (longitudinal["subject_id"] == sid)
            & (longitudinal["tumor_size"].notna())
        ].sort_values("days_since_randomization")

        if subj_tumors.empty:
            # No tumor data: use OS as PFS
            results.append({"subject_id": sid, "pfs_time": os_time, "pfs_event": os_event})
            continue

        tumors = subj_tumors[["days_since_randomization", "tumor_size"]].values
        nadir = tumors[0, 1]
        progression_day = None

        for day, size in tumors[1:]:
            nadir = min(nadir, size)
            if nadir > 0 and size >= nadir * 1.20:
                progression_day = day
                break

        if progression_day is not None:
            pfs_time = min(progression_day, os_time)
            pfs_event = 1
        elif os_event == 1:
            pfs_time = os_time
            pfs_event = 1
        else:
            last_assess = tumors[-1, 0]
            pfs_time = min(last_assess, os_time)
            pfs_event = 0

        results.append({"subject_id": sid, "pfs_time": pfs_time, "pfs_event": int(pfs_event)})

    return pd.DataFrame(results)
