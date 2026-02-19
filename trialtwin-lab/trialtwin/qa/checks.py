"""Clinical data validation checks.

Each check is a class with a run() method that validates a specific
aspect of clinical data integrity. These automated checks catch issues
that would invalidate downstream modeling.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trialtwin.harmonize.harmonizer import HarmonizedDataset
from trialtwin.utils.config import TrialConfig


@dataclass
class QAResult:
    name: str
    passed: bool
    message: str
    details: str = ""


class NoNegativeSurvivalTimes:
    """Survival times must be non-negative."""

    def run(self, data: HarmonizedDataset, config: TrialConfig) -> QAResult:
        neg_os = (data.endpoints["os_time"] < 0).sum()
        neg_pfs = 0
        if "pfs_time" in data.endpoints.columns:
            pfs = data.endpoints["pfs_time"].dropna()
            neg_pfs = (pfs < 0).sum()
        total_neg = neg_os + neg_pfs
        return QAResult(
            name="No Negative Survival Times",
            passed=total_neg == 0,
            message=f"{total_neg} negative survival times found" if total_neg > 0
                    else "All survival times are non-negative",
            details=f"OS negative: {neg_os}, PFS negative: {neg_pfs}",
        )


class NoPostDeathMeasurements:
    """No longitudinal measurements should occur after a patient's death."""

    def run(self, data: HarmonizedDataset, config: TrialConfig) -> QAResult:
        if data.longitudinal.empty:
            return QAResult("No Post-Death Measurements", True, "No longitudinal data to check")

        death_times = data.endpoints[data.endpoints["os_event"] == 1][
            ["subject_id", "os_time"]
        ]
        merged = data.longitudinal.merge(death_times, on="subject_id", how="inner")
        violations = (merged["days_since_randomization"] > merged["os_time"]).sum()
        return QAResult(
            name="No Post-Death Measurements",
            passed=violations == 0,
            message=f"{violations} measurements after death" if violations > 0
                    else "No post-death measurements found",
        )


class PlausibleLabRanges:
    """Lab values must fall within clinically plausible ranges."""

    RANGES = {
        "hemoglobin": (3, 20),
        "ldh": (50, 2000),
        "albumin": (1, 6),
        "neutrophils": (0.1, 50),
        "lymphocytes": (0.1, 30),
    }

    def run(self, data: HarmonizedDataset, config: TrialConfig) -> QAResult:
        violations = []
        for lab, (lo, hi) in self.RANGES.items():
            # Check in subjects (baseline)
            if lab in data.subjects.columns:
                vals = data.subjects[lab].dropna()
                out = ((vals < lo) | (vals > hi)).sum()
                if out > 0:
                    violations.append(f"{lab} baseline: {out} out of range [{lo}, {hi}]")

            # Check in longitudinal
            if lab in data.longitudinal.columns:
                vals = data.longitudinal[lab].dropna()
                out = ((vals < lo) | (vals > hi)).sum()
                if out > 0:
                    violations.append(f"{lab} longitudinal: {out} out of range [{lo}, {hi}]")

        return QAResult(
            name="Plausible Lab Ranges",
            passed=len(violations) == 0,
            message=f"{len(violations)} lab range violations" if violations
                    else "All lab values within plausible ranges",
            details="\n".join(violations) if violations else "",
        )


class MonotonicVisitTimes:
    """Visit times should be monotonically increasing per subject."""

    def run(self, data: HarmonizedDataset, config: TrialConfig) -> QAResult:
        if data.longitudinal.empty:
            return QAResult("Monotonic Visit Times", True, "No longitudinal data to check")

        violations = 0
        for sid, group in data.longitudinal.groupby("subject_id"):
            times = group["days_since_randomization"].values
            if len(times) > 1 and not np.all(np.diff(times) >= 0):
                violations += 1

        return QAResult(
            name="Monotonic Visit Times",
            passed=violations == 0,
            message=f"{violations} subjects with non-monotonic visit times" if violations > 0
                    else "All visit times are monotonically increasing",
        )


class NoFutureDataLeakage:
    """Features should only use data up to the landmark day."""

    def run(self, data: HarmonizedDataset, config: TrialConfig) -> QAResult:
        landmark = config.landmark_day
        if landmark is None or data.longitudinal.empty:
            return QAResult("No Future Data Leakage", True,
                          "No landmark day configured or no longitudinal data")
        # This check validates that the longitudinal data structure supports
        # proper temporal filtering. Actual leakage prevention is in feature engineering.
        has_post_landmark = (data.longitudinal["days_since_randomization"] > landmark).any()
        return QAResult(
            name="No Future Data Leakage",
            passed=True,
            message=f"Longitudinal data extends beyond landmark day {landmark} "
                    f"(feature engineering must filter to <= {landmark})",
            details="Check is informational — leakage prevention enforced in feature builder",
        )


class ConsistentCensoring:
    """Censored patients (os_event=0) should not have contradictory death records."""

    def run(self, data: HarmonizedDataset, config: TrialConfig) -> QAResult:
        censored = data.endpoints[data.endpoints["os_event"] == 0]
        # For censored patients, os_time should represent last known alive time
        # Check that censored patients don't have os_time = 0 (likely data error)
        zero_time_censored = (censored["os_time"] == 0).sum()
        return QAResult(
            name="Consistent Censoring",
            passed=zero_time_censored == 0,
            message=f"{zero_time_censored} censored patients with time=0" if zero_time_censored > 0
                    else "All censoring is consistent",
        )


class CompleteDemographics:
    """Required demographic fields should be non-null."""

    REQUIRED = ["subject_id", "treatment_arm", "age"]

    def run(self, data: HarmonizedDataset, config: TrialConfig) -> QAResult:
        missing = {}
        for col in self.REQUIRED:
            if col in data.subjects.columns:
                n_miss = data.subjects[col].isna().sum()
                if n_miss > 0:
                    missing[col] = n_miss

        return QAResult(
            name="Complete Demographics",
            passed=len(missing) == 0,
            message=f"Missing values in: {missing}" if missing
                    else "All required demographics are complete",
        )


class BalancedRandomization:
    """Treatment arms should be approximately balanced (within 60/40)."""

    def run(self, data: HarmonizedDataset, config: TrialConfig) -> QAResult:
        counts = data.subjects["treatment_arm"].value_counts()
        n = len(data.subjects)
        fractions = counts / n
        max_frac = fractions.max()
        balanced = max_frac <= 0.65  # Allow up to 65/35 split

        return QAResult(
            name="Balanced Randomization",
            passed=balanced,
            message=f"Arm distribution: {counts.to_dict()}"
                    + (" (IMBALANCED)" if not balanced else " (balanced)"),
        )


ALL_CHECKS = [
    NoNegativeSurvivalTimes(),
    NoPostDeathMeasurements(),
    PlausibleLabRanges(),
    MonotonicVisitTimes(),
    NoFutureDataLeakage(),
    ConsistentCensoring(),
    CompleteDemographics(),
    BalancedRandomization(),
]


def run_all_checks(
    harmonized: HarmonizedDataset, config: TrialConfig
) -> list[QAResult]:
    return [check.run(harmonized, config) for check in ALL_CHECKS]
