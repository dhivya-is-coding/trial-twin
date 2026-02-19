"""Baseline feature extraction from subject demographics and labs."""

import numpy as np
import pandas as pd


def extract_baseline_features(subjects: pd.DataFrame) -> pd.DataFrame:
    """Extract baseline features from subject demographics and labs.

    Returns one row per subject with numeric features only
    (plus subject_id for merging).
    """
    features = pd.DataFrame({"subject_id": subjects["subject_id"]})

    # Demographics
    features["age"] = subjects["age"]
    features["sex_male"] = (subjects["sex"] == "M").astype(int)

    if "ecog" in subjects.columns:
        features["ecog"] = subjects["ecog"].fillna(1)  # impute median ECOG

    # Tumor burden (log-transform for log-normal distribution)
    if "tumor_burden" in subjects.columns:
        features["log_tumor_burden"] = np.log1p(subjects["tumor_burden"])

    # Labs
    if "hemoglobin" in subjects.columns:
        features["hemoglobin"] = subjects["hemoglobin"]
        features["anemia_flag"] = (subjects["hemoglobin"] < 10).astype(int)

    if "ldh" in subjects.columns:
        features["log_ldh"] = np.log1p(subjects["ldh"])

    if "albumin" in subjects.columns:
        features["albumin"] = subjects["albumin"]
        features["hypoalbuminemia_flag"] = (subjects["albumin"] < 3.5).astype(int)

    if "neutrophils" in subjects.columns:
        features["log_neutrophils"] = np.log1p(subjects["neutrophils"])

    # Neutrophil-to-lymphocyte ratio
    if "neutrophils" in subjects.columns and "lymphocytes" in subjects.columns:
        features["nlr"] = subjects["neutrophils"] / subjects["lymphocytes"].clip(lower=0.1)

    # GBSG2-specific features
    if "positive_nodes" in subjects.columns:
        features["log_positive_nodes"] = np.log1p(subjects["positive_nodes"])

    if "progesterone_receptor" in subjects.columns:
        features["log_progrec"] = np.log1p(subjects["progesterone_receptor"])

    if "estrogen_receptor" in subjects.columns:
        features["log_estrec"] = np.log1p(subjects["estrogen_receptor"])

    return features
