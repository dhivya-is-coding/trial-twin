"""YAML configuration loader for trial definitions."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrialConfig:
    """Trial configuration loaded from YAML."""

    raw: dict[str, Any]
    config_path: Path

    @classmethod
    def load(cls, path: str | Path) -> "TrialConfig":
        path = Path(path)
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(raw=raw, config_path=path)

    @property
    def trial_id(self) -> str:
        return self.raw["trial"]["id"]

    @property
    def trial_name(self) -> str:
        return self.raw["trial"]["name"]

    @property
    def disease(self) -> str:
        return self.raw["trial"]["disease"]

    @property
    def source(self) -> str:
        return self.raw["trial"]["source"]

    @property
    def n_subjects(self) -> int | None:
        return self.raw["trial"].get("n_subjects")

    @property
    def seed(self) -> int:
        return self.raw["trial"].get("seed", 42)

    @property
    def arms(self) -> dict[str, str]:
        return self.raw["trial"]["arms"]

    @property
    def control_arm(self) -> str:
        return self.raw["trial"]["arms"]["control"]

    @property
    def treatment_arm(self) -> str:
        return self.raw["trial"]["arms"]["treatment"]

    @property
    def landmark_day(self) -> int | None:
        return self.raw.get("features", {}).get("landmark_day")

    @property
    def has_longitudinal(self) -> bool:
        return self.raw.get("features", {}).get("has_longitudinal", True)

    @property
    def endpoints(self) -> dict:
        return self.raw.get("endpoints", {"os": True, "pfs": False})

    def get(self, *keys: str, default: Any = None) -> Any:
        """Nested dict access: config.get('survival', 'os_shape')"""
        d = self.raw
        for k in keys:
            if isinstance(d, dict):
                d = d.get(k, default)
            else:
                return default
        return d
