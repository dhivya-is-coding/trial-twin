# TrialTwin Lab

End-to-end digital twin pipeline for oncology clinical trials. Generates counterfactual control-arm predictions using baseline prognostic models, then quantifies the statistical efficiency gain from PROCOVA-style covariate adjustment.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
make install
make all        # generates data, trains models, produces figures
make app        # launches interactive Streamlit dashboard
```

## What This Does

```
YAML Config ──> Data Ingest ──> Harmonize ──> QA Checks (8/8)
                                                    │
                Endpoint Derivation (OS, PFS, Landmark)
                                                    │
                Feature Engineering (baseline + longitudinal)
                                                    │
            ┌── Cox PH (baseline) ──> Digital Twins ──> Prognostic Score
            │   GBM (all features)                           │
            │                                      PROCOVA Efficiency Sim
            │                                                │
            └── Streamlit Dashboard ─────────────────────────┘
```

1. **Data Ingest** — Synthetic NSCLC generator (300 patients, Weibull survival, correlated covariates) or GBSG2 real breast cancer data (686 patients)
2. **Harmonization + QA** — Unified schema with 8 automated clinical data checks
3. **Endpoints** — Overall survival, progression-free survival (RECIST-simplified), landmark binary
4. **Features** — Baseline demographics/labs + 60-day longitudinal tumor dynamics and lab trajectories
5. **Modeling** — Cox PH (baseline features, C-index 0.62) and gradient boosting survival (all features, C-index 0.90) on control arm
6. **Digital Twins** — Predict each treated patient's counterfactual control outcome; estimate individual treatment effects
7. **Efficiency** — PROCOVA-style analysis: adjusting for prognostic score reduces 95% CI width by ~8%, meaning fewer patients needed for the same power

## Key Results

| Metric | Value |
|--------|-------|
| Subjects | 300 (157 control, 143 treatment) |
| Cox C-index (baseline) | 0.615 |
| GBM C-index (all features) | 0.901 |
| Standard HR | 0.771 (0.603–0.987) |
| Adjusted HR (PROCOVA) | 0.703 (0.548–0.902) |
| CI Width Reduction | 7.8% |
| Mean Treatment Effect | +89 days |

## Design Decisions

- **Config-driven**: YAML defines everything — switch disease area by swapping configs
- **QA-first**: 8 automated checks run before any modeling (no negative times, no post-death measurements, plausible lab ranges, balanced arms, etc.)
- **Baseline-only prognostic model**: Digital twins use only pre-treatment features to avoid confounding longitudinal features with treatment response
- **Production structure**: Proper Python package with CLI, editable install, Makefile targets
- **Extensible**: Add new data sources by implementing `DataSource.load() -> dict[str, DataFrame]`

## Project Structure

```
trialtwin-lab/
├── configs/
│   ├── trial_nsclc.yaml          # Synthetic NSCLC (primary)
│   └── trial_gbsg2.yaml          # Real breast cancer data
├── trialtwin/
│   ├── ingest/                   # Data sources (synthetic, GBSG2)
│   ├── harmonize/                # Schema normalization + parquet I/O
│   ├── qa/                       # 8 clinical data validation checks
│   ├── endpoints/                # OS, PFS, landmark derivation
│   ├── features/                 # Baseline + longitudinal feature engineering
│   ├── model/                    # Cox PH, GBM survival, digital twin generator
│   ├── efficiency/               # PROCOVA-style simulation
│   └── utils/                    # Config loader, plotting
├── scripts/
│   └── run_pipeline.py           # CLI orchestrator (click)
├── app/
│   ├── streamlit_app.py          # Dashboard entry point
│   └── pages/                    # Patient Explorer + Trial Dashboard
├── tests/                        # 14 tests (endpoints, features, QA)
├── Makefile                      # install, data, train, report, app, test, clean
└── outputs/
    ├── figures/                  # KM curves, efficiency comparison
    └── reports/                  # QA report (markdown)
```

## Built With

- **lifelines** — Cox PH modeling, Kaplan-Meier curves
- **scikit-survival** — Gradient boosting survival analysis, GBSG2 dataset
- **Streamlit + Plotly** — Interactive dashboard
- **pandas / numpy / scipy** — Data processing
- **click** — CLI orchestration
