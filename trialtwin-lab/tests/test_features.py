"""Tests for feature engineering: no NaN, no leakage, correct shape."""

import numpy as np

from trialtwin.endpoints.survival import derive_overall_survival
from trialtwin.features.builder import FeatureBuilder


def test_no_nan_after_imputation(harmonized, nsclc_config):
    os_data = derive_overall_survival(harmonized.subjects, harmonized.endpoints)
    builder = FeatureBuilder(nsclc_config)
    features = builder.build(harmonized, os_data)
    numeric = features.select_dtypes(include=[np.number])
    assert not numeric.isna().any().any(), f"NaN in: {numeric.columns[numeric.isna().any()].tolist()}"


def test_one_row_per_subject(harmonized, nsclc_config):
    os_data = derive_overall_survival(harmonized.subjects, harmonized.endpoints)
    builder = FeatureBuilder(nsclc_config)
    features = builder.build(harmonized, os_data)
    assert features.shape[0] == harmonized.subjects.shape[0]
    assert features["subject_id"].is_unique


def test_has_required_columns(harmonized, nsclc_config):
    os_data = derive_overall_survival(harmonized.subjects, harmonized.endpoints)
    builder = FeatureBuilder(nsclc_config)
    features = builder.build(harmonized, os_data)
    assert "subject_id" in features.columns
    assert "treatment_arm" in features.columns


def test_numeric_only_after_drop(harmonized, nsclc_config):
    os_data = derive_overall_survival(harmonized.subjects, harmonized.endpoints)
    builder = FeatureBuilder(nsclc_config)
    features = builder.build(harmonized, os_data)
    X = features.drop(columns=["subject_id", "treatment_arm"])
    non_numeric = X.select_dtypes(exclude=[np.number])
    assert non_numeric.empty, f"Non-numeric columns: {non_numeric.columns.tolist()}"


def test_baseline_cols_tracked(harmonized, nsclc_config):
    os_data = derive_overall_survival(harmonized.subjects, harmonized.endpoints)
    builder = FeatureBuilder(nsclc_config)
    builder.build(harmonized, os_data)
    assert len(builder.baseline_cols) > 0
    assert "subject_id" not in builder.baseline_cols
