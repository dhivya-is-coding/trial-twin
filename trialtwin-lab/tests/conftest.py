"""Shared fixtures for TrialTwin tests."""

import pytest
from pathlib import Path

from trialtwin.utils.config import TrialConfig
from trialtwin.ingest.synthetic import SyntheticSource
from trialtwin.harmonize.harmonizer import Harmonizer


BASE_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def nsclc_config():
    return TrialConfig.load(BASE_DIR / "configs" / "trial_nsclc.yaml")


@pytest.fixture(scope="session")
def raw_data(nsclc_config):
    source = SyntheticSource(nsclc_config)
    return source.load()


@pytest.fixture(scope="session")
def harmonized(nsclc_config, raw_data):
    harmonizer = Harmonizer(nsclc_config)
    return harmonizer.harmonize(raw_data)
