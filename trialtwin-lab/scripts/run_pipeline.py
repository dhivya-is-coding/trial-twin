"""CLI orchestrator for the TrialTwin pipeline."""

import json
import pickle
from pathlib import Path

import click

from trialtwin.utils.config import TrialConfig

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"


def _load_data(config: TrialConfig) -> dict:
    """Load data from the configured source."""
    if config.source == "synthetic":
        from trialtwin.ingest.synthetic import SyntheticSource
        source = SyntheticSource(config)
    elif config.source == "gbsg2":
        from trialtwin.ingest.gbsg2_loader import GBSG2Source
        source = GBSG2Source(config)
    else:
        raise ValueError(f"Unknown source: {config.source}")
    return source.load()


@click.group()
def cli():
    """TrialTwin Oncology Lab - Digital twins for clinical trials."""
    pass


@cli.command()
@click.option("--config", required=True, help="Path to trial YAML config")
def data(config: str):
    """Generate or load trial data and save harmonized dataset."""
    cfg = TrialConfig.load(config)
    click.echo(f"Loading data for trial: {cfg.trial_name}")

    raw = _load_data(cfg)

    from trialtwin.harmonize.harmonizer import Harmonizer
    harmonizer = Harmonizer(cfg)
    harmonized = harmonizer.harmonize(raw)

    from trialtwin.qa.checks import run_all_checks
    from trialtwin.qa.report import generate_qa_report
    results = run_all_checks(harmonized, cfg)
    report_path = OUTPUT_DIR / "reports" / "qa_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    generate_qa_report(results, report_path)
    click.echo(f"QA report saved to {report_path}")

    out = DATA_DIR / "processed" / cfg.trial_id
    out.mkdir(parents=True, exist_ok=True)
    harmonized.to_parquet(out)
    click.echo(f"Harmonized data saved to {out}")
    harmonized.summary()


@cli.command()
@click.option("--config", required=True, help="Path to trial YAML config")
def train(config: str):
    """Run endpoint derivation, feature engineering, modeling, and digital twins."""
    cfg = TrialConfig.load(config)
    click.echo(f"Training pipeline for: {cfg.trial_name}")

    from trialtwin.harmonize.harmonizer import HarmonizedDataset
    data_dir = DATA_DIR / "processed" / cfg.trial_id
    harmonized = HarmonizedDataset.from_parquet(data_dir)

    # Endpoint derivation
    from trialtwin.endpoints.survival import derive_overall_survival, derive_pfs
    from trialtwin.endpoints.binary import derive_landmark_survival
    os_data = derive_overall_survival(harmonized.subjects, harmonized.endpoints)
    click.echo(f"OS derived: {os_data.shape[0]} subjects, {os_data['os_event'].sum():.0f} events")

    pfs_data = None
    if cfg.endpoints.get("pfs", False) and cfg.has_longitudinal:
        pfs_data = derive_pfs(harmonized.subjects, harmonized.longitudinal, harmonized.endpoints)
        click.echo(f"PFS derived: {pfs_data.shape[0]} subjects, {pfs_data['pfs_event'].sum():.0f} events")

    landmark = derive_landmark_survival(os_data, cfg.get("endpoints", "binary_landmark_days", default=365))
    click.echo(f"12-month survival rate: {landmark['survived_landmark'].mean():.1%}")

    # Feature engineering
    from trialtwin.features.builder import FeatureBuilder
    builder = FeatureBuilder(cfg)
    features = builder.build(harmonized, os_data)
    click.echo(f"Feature matrix: {features.shape[0]} subjects x {features.shape[1]} features")

    # Modeling (control arm only)
    # Baseline-only features for prognostic model (digital twins)
    # Longitudinal features capture treatment response, so they can't be used
    # for counterfactual predictions on treated patients.
    from trialtwin.model.cox import CoxPHModel
    from trialtwin.model.evaluate import evaluate_model, save_evaluation_plots

    baseline_cols = builder.baseline_cols
    control_mask = features["treatment_arm"] == "control"
    treated_mask = features["treatment_arm"] == "treatment"

    X_control_baseline = features.loc[control_mask, baseline_cols]
    X_control_all = features[control_mask].drop(columns=["subject_id", "treatment_arm"])
    os_control = os_data[control_mask]

    # Prognostic model (baseline only) — used for digital twins and efficiency
    cox = CoxPHModel(penalizer=0.1)
    cox.fit(X_control_baseline, os_control["os_time"], os_control["os_event"])
    click.echo("\nCox PH Prognostic Model (baseline features):")
    click.echo(cox.summary().to_string())

    metrics = evaluate_model(cox, X_control_baseline, os_control)
    click.echo(f"\nControl-arm C-index (baseline): {metrics['c_index']:.3f}")

    fig_dir = OUTPUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    save_evaluation_plots(cox, X_control_baseline, os_control, fig_dir)

    # GBM model (uses all features — tree models handle collinearity)
    from trialtwin.model.gbm import GBMSurvivalModel
    gbm = GBMSurvivalModel()
    gbm.fit(X_control_all, os_control["os_time"], os_control["os_event"])
    gbm_metrics = evaluate_model(gbm, X_control_all, os_control)
    click.echo(f"GBM C-index (all features): {gbm_metrics['c_index']:.3f}")

    # Digital twins — apply baseline prognostic model to treated patients
    from trialtwin.model.digital_twin import DigitalTwinGenerator
    dtg = DigitalTwinGenerator(cox)

    X_treated_baseline = features.loc[treated_mask, baseline_cols]
    os_treated = os_data[treated_mask]

    twins = dtg.generate_twins(X_treated_baseline, features[treated_mask]["subject_id"])
    effects = dtg.estimate_individual_treatment_effect(twins, os_treated)
    click.echo(f"\nDigital twins generated for {len(twins)} treated subjects")
    click.echo(f"Mean predicted control median: {twins['predicted_median_survival'].mean():.0f} days")
    click.echo(f"Mean individual treatment effect: {effects['treatment_effect'].mean():.0f} days")

    # Efficiency simulation — prognostic score from baseline model
    from trialtwin.efficiency.simulation import run_efficiency_simulation
    all_baseline = features[baseline_cols]
    prognostic_scores = dtg.compute_prognostic_score(all_baseline)
    efficiency = run_efficiency_simulation(
        os_data, features["treatment_arm"], prognostic_scores
    )
    click.echo(f"\n--- Efficiency Results ---")
    click.echo(f"Standard log-rank HR: {efficiency['standard_hr']:.3f} "
               f"(95% CI: {efficiency['standard_ci'][0]:.3f}-{efficiency['standard_ci'][1]:.3f})")
    click.echo(f"Adjusted HR: {efficiency['adjusted_hr']:.3f} "
               f"(95% CI: {efficiency['adjusted_ci'][0]:.3f}-{efficiency['adjusted_ci'][1]:.3f})")
    click.echo(f"CI width reduction: {efficiency['ci_width_reduction_pct']:.1f}%")

    # Save artifacts
    artifacts = {
        "cox_model": cox,
        "gbm_model": gbm,
        "features": features,
        "os_data": os_data,
        "pfs_data": pfs_data,
        "twins": twins,
        "effects": effects,
        "efficiency": efficiency,
        "prognostic_scores": prognostic_scores,
        "metrics": {"cox_c_index": metrics["c_index"], "gbm_c_index": gbm_metrics["c_index"]},
    }
    artifact_path = DATA_DIR / "outputs" / cfg.trial_id / "artifacts.pkl"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with open(artifact_path, "wb") as f:
        pickle.dump(artifacts, f)
    click.echo(f"\nArtifacts saved to {artifact_path}")


@cli.command()
@click.option("--config", required=True, help="Path to trial YAML config")
def report(config: str):
    """Generate final report with all figures and summaries."""
    cfg = TrialConfig.load(config)
    click.echo(f"Generating report for: {cfg.trial_name}")

    artifact_path = DATA_DIR / "outputs" / cfg.trial_id / "artifacts.pkl"
    with open(artifact_path, "rb") as f:
        artifacts = pickle.load(f)

    from trialtwin.utils.plotting import generate_all_plots
    fig_dir = OUTPUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    generate_all_plots(artifacts, cfg, fig_dir)
    click.echo(f"Figures saved to {fig_dir}")


if __name__ == "__main__":
    cli()
