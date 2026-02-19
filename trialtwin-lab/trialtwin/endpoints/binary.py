"""Binary landmark survival endpoint."""

import pandas as pd


def derive_landmark_survival(
    os_data: pd.DataFrame, landmark_days: int = 365
) -> pd.DataFrame:
    """Derive binary landmark survival indicator.

    survived_landmark = 1 if patient survived beyond landmark_days
    (either still alive at landmark or censored after landmark).
    """
    result = os_data[["subject_id"]].copy()
    result["survived_landmark"] = (
        (os_data["os_time"] >= landmark_days)
        | ((os_data["os_event"] == 0) & (os_data["os_time"] >= landmark_days))
    ).astype(int)
    return result
