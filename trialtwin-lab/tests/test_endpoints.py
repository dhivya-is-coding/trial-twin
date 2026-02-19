"""Tests for endpoint derivation: OS, PFS, and landmark survival."""

import numpy as np

from trialtwin.endpoints.survival import derive_overall_survival, derive_pfs
from trialtwin.endpoints.binary import derive_landmark_survival


def test_os_no_negative_times(harmonized):
    os_data = derive_overall_survival(harmonized.subjects, harmonized.endpoints)
    assert (os_data["os_time"] >= 0).all(), "Found negative OS times"


def test_os_events_binary(harmonized):
    os_data = derive_overall_survival(harmonized.subjects, harmonized.endpoints)
    assert set(os_data["os_event"].unique()).issubset({0, 1}), "OS events not binary"


def test_os_has_treatment_arm(harmonized):
    os_data = derive_overall_survival(harmonized.subjects, harmonized.endpoints)
    assert "treatment_arm" in os_data.columns
    assert set(os_data["treatment_arm"].unique()) == {"control", "treatment"}


def test_pfs_no_negative_times(harmonized, nsclc_config):
    os_data = derive_overall_survival(harmonized.subjects, harmonized.endpoints)
    pfs_data = derive_pfs(harmonized.subjects, harmonized.longitudinal, harmonized.endpoints)
    assert (pfs_data["pfs_time"] >= 0).all(), "Found negative PFS times"


def test_pfs_leq_os(harmonized, nsclc_config):
    os_data = derive_overall_survival(harmonized.subjects, harmonized.endpoints)
    pfs_data = derive_pfs(harmonized.subjects, harmonized.longitudinal, harmonized.endpoints)
    merged = os_data[["subject_id", "os_time"]].merge(
        pfs_data[["subject_id", "pfs_time"]], on="subject_id"
    )
    violations = (merged["pfs_time"] > merged["os_time"] + 1.0).sum()  # +1 for float tolerance
    assert violations == 0, f"PFS > OS for {violations} subjects"


def test_landmark_survival_plausible(harmonized):
    os_data = derive_overall_survival(harmonized.subjects, harmonized.endpoints)
    landmark = derive_landmark_survival(os_data, landmark_days=365)
    rate = landmark["survived_landmark"].mean()
    assert 0.1 < rate < 0.9, f"Landmark survival rate {rate:.2f} outside plausible range"
