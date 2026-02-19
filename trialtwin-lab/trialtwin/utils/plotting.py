"""Shared plotting utilities for KM curves, calibration, and reports."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

from trialtwin.utils.config import TrialConfig


def generate_all_plots(artifacts: dict, config: TrialConfig, fig_dir: Path) -> None:
    """Generate all publication-quality figures from pipeline artifacts."""
    fig_dir.mkdir(parents=True, exist_ok=True)

    os_data = artifacts["os_data"]
    features = artifacts["features"]
    efficiency = artifacts["efficiency"]
    metrics = artifacts["metrics"]

    plot_km_by_arm(os_data, config, fig_dir)
    plot_efficiency_comparison(efficiency, fig_dir)


def plot_km_by_arm(os_data: pd.DataFrame, config: TrialConfig, fig_dir: Path) -> None:
    """Plot Kaplan-Meier survival curves by treatment arm."""
    fig, ax = plt.subplots(figsize=(8, 6))
    kmf = KaplanMeierFitter()

    control = os_data[os_data["treatment_arm"] == "control"]
    treated = os_data[os_data["treatment_arm"] == "treatment"]

    kmf.fit(control["os_time"], event_observed=control["os_event"], label=config.control_arm)
    kmf.plot_survival_function(ax=ax)

    kmf.fit(treated["os_time"], event_observed=treated["os_event"], label=config.treatment_arm)
    kmf.plot_survival_function(ax=ax)

    # Log-rank test
    lr = logrank_test(
        control["os_time"], treated["os_time"],
        event_observed_A=control["os_event"], event_observed_B=treated["os_event"],
    )

    ax.set_xlabel("Days from Randomization")
    ax.set_ylabel("Overall Survival Probability")
    ax.set_title(f"Kaplan-Meier: {config.trial_name}")
    ax.text(0.6, 0.9, f"Log-rank p = {lr.p_value:.4f}", transform=ax.transAxes)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "km_by_arm.png", dpi=150)
    plt.close(fig)


def plot_efficiency_comparison(efficiency: dict, fig_dir: Path) -> None:
    """Bar chart comparing standard vs adjusted CI width."""
    standard_width = efficiency["standard_ci"][1] - efficiency["standard_ci"][0]
    adjusted_width = efficiency["adjusted_ci"][1] - efficiency["adjusted_ci"][0]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        ["Standard\n(Log-rank)", "PROCOVA-style\n(Adjusted)"],
        [standard_width, adjusted_width],
        color=["#4a90d9", "#45b39d"],
        width=0.5,
    )

    ax.set_ylabel("95% CI Width (HR)")
    ax.set_title("Treatment Effect Precision: Standard vs Adjusted")
    pct = efficiency["ci_width_reduction_pct"]
    ax.text(0.5, 0.95, f"{pct:.1f}% CI width reduction",
            transform=ax.transAxes, ha="center", fontweight="bold")

    for bar, val in zip(bars, [standard_width, adjusted_width]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig(fig_dir / "efficiency_comparison.png", dpi=150)
    plt.close(fig)
