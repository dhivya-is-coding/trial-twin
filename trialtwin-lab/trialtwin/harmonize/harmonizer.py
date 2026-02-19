"""Config-driven data harmonization to unified schema.

Maps raw trial data from any source (synthetic, GBSG2, PDS) to a
standardized three-table schema: subjects, longitudinal, endpoints.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from trialtwin.utils.config import TrialConfig


@dataclass
class HarmonizedDataset:
    """Unified trial dataset with three core tables."""

    subjects: pd.DataFrame
    longitudinal: pd.DataFrame
    endpoints: pd.DataFrame

    def to_parquet(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.subjects.to_parquet(path / "subjects.parquet", index=False)
        self.longitudinal.to_parquet(path / "longitudinal.parquet", index=False)
        self.endpoints.to_parquet(path / "endpoints.parquet", index=False)

    @classmethod
    def from_parquet(cls, path: Path | str) -> "HarmonizedDataset":
        path = Path(path)
        return cls(
            subjects=pd.read_parquet(path / "subjects.parquet"),
            longitudinal=pd.read_parquet(path / "longitudinal.parquet"),
            endpoints=pd.read_parquet(path / "endpoints.parquet"),
        )

    def summary(self) -> None:
        n = len(self.subjects)
        arms = self.subjects["treatment_arm"].value_counts().to_dict()
        os_events = self.endpoints["os_event"].sum()
        n_visits = len(self.longitudinal)
        print(f"Harmonized dataset: {n} subjects")
        print(f"  Arms: {arms}")
        print(f"  OS events: {int(os_events)} / {n} ({os_events/n:.1%})")
        print(f"  Longitudinal records: {n_visits}")
        if n_visits > 0:
            n_with_long = self.longitudinal["subject_id"].nunique()
            print(f"  Subjects with longitudinal data: {n_with_long}")


class Harmonizer:
    """Map raw trial data to the unified HarmonizedDataset schema."""

    def __init__(self, config: TrialConfig):
        self.config = config

    def harmonize(self, raw: dict[str, pd.DataFrame]) -> HarmonizedDataset:
        subjects = raw["subjects"].copy()
        longitudinal = raw["longitudinal"].copy()
        endpoints = raw["endpoints"].copy()

        # Ensure required columns exist
        required_subject_cols = ["subject_id", "treatment_arm", "age"]
        for col in required_subject_cols:
            if col not in subjects.columns:
                raise ValueError(f"Missing required column in subjects: {col}")

        required_endpoint_cols = ["subject_id", "os_time", "os_event"]
        for col in required_endpoint_cols:
            if col not in endpoints.columns:
                raise ValueError(f"Missing required column in endpoints: {col}")

        # Ensure consistent types
        endpoints["os_event"] = endpoints["os_event"].astype(int)
        if "pfs_event" in endpoints.columns:
            endpoints["pfs_event"] = endpoints["pfs_event"].fillna(0).astype(int)
            endpoints["pfs_time"] = endpoints["pfs_time"].fillna(endpoints["os_time"])

        return HarmonizedDataset(
            subjects=subjects,
            longitudinal=longitudinal,
            endpoints=endpoints,
        )
