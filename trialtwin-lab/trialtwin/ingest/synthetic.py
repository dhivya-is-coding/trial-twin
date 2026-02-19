"""Synthetic oncology clinical trial data generator.

Generates clinically realistic NSCLC trial data with proper correlation
structures: sicker patients (high ECOG, elevated LDH, low albumin) have
worse survival outcomes, matching real oncology data patterns.

Outputs data in CDISC SDTM-like format and also saves raw domain CSVs.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from trialtwin.ingest.base import DataSource
from trialtwin.utils.config import TrialConfig


class SyntheticSource(DataSource):
    """Config-driven synthetic clinical trial data generator."""

    def __init__(self, config: TrialConfig, output_dir: Path | None = None):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.output_dir = output_dir

    def load(self) -> dict[str, pd.DataFrame]:
        n = self.config.n_subjects
        subjects = self._generate_subjects(n)
        endpoints = self._generate_survival(subjects)
        longitudinal = self._generate_longitudinal(subjects, endpoints)
        endpoints = self._derive_pfs(longitudinal, endpoints)

        if self.output_dir:
            self._save_sdtm_csvs(subjects, longitudinal, endpoints)

        return {
            "subjects": subjects,
            "longitudinal": longitudinal,
            "endpoints": endpoints,
        }

    def _generate_subjects(self, n: int) -> pd.DataFrame:
        """Generate baseline demographics and labs with realistic correlations."""
        cfg = self.config

        # Demographics
        ages = self.rng.normal(
            cfg.get("demographics", "age", "mean"),
            cfg.get("demographics", "age", "std"),
            size=n,
        ).clip(
            cfg.get("demographics", "age", "min"),
            cfg.get("demographics", "age", "max"),
        )

        sexes = self.rng.choice(
            ["M", "F"],
            size=n,
            p=[cfg.get("demographics", "male_fraction"),
               1 - cfg.get("demographics", "male_fraction")],
        )

        ecog_weights = cfg.get("demographics", "ecog_weights")
        ecog = self.rng.choice([0, 1, 2], size=n, p=ecog_weights)

        race_map = cfg.get("demographics", "race_weights")
        races = self.rng.choice(
            list(race_map.keys()), size=n, p=list(race_map.values())
        )

        # Treatment assignment
        ratio = cfg.get("trial", "randomization_ratio")
        p_treat = ratio[1] / sum(ratio)
        arms = self.rng.choice(
            ["control", "treatment"], size=n, p=[1 - p_treat, p_treat]
        )

        # Baseline tumor burden (log-normal, correlated with ECOG)
        tumor_cfg = cfg.get("tumor", "baseline_burden")
        log_tumor = self.rng.normal(
            tumor_cfg["mean"] + 0.2 * ecog,  # sicker patients have larger tumors
            tumor_cfg["std"],
            size=n,
        )
        tumor_burden = np.exp(log_tumor)

        # Baseline labs with correlation structure
        # Sicker patients: lower hemoglobin, lower albumin, higher LDH
        sickness_factor = 0.3 * ecog + 0.1 * (log_tumor - tumor_cfg["mean"])

        labs = {}
        for lab_name, lab_cfg in cfg.get("baseline_labs").items():
            mean = lab_cfg["mean"]
            std = lab_cfg["std"]

            # Apply sickness correlation
            if lab_name in ("hemoglobin", "albumin", "lymphocytes"):
                adjusted_mean = mean - sickness_factor * std * 0.5
            elif lab_name in ("ldh", "neutrophils"):
                adjusted_mean = mean + sickness_factor * std * 0.5
            else:
                adjusted_mean = mean

            if lab_cfg.get("log_normal", False):
                log_val = self.rng.normal(np.log(adjusted_mean), std / mean, size=n)
                labs[lab_name] = np.exp(log_val)
            else:
                labs[lab_name] = self.rng.normal(adjusted_mean, std, size=n)

            # Clip to plausible ranges
            nrange = lab_cfg.get("normal_range", [0, np.inf])
            labs[lab_name] = np.clip(labs[lab_name], nrange[0] * 0.25, nrange[1] * 1.15)

        # Randomization dates (spread over 12 months for realism)
        rand_days_offset = self.rng.integers(0, 365, size=n)
        base_date = pd.Timestamp("2020-01-01")
        rand_dates = [base_date + pd.Timedelta(days=int(d)) for d in rand_days_offset]

        subjects = pd.DataFrame({
            "subject_id": [f"{cfg.trial_id}-{i:04d}" for i in range(n)],
            "trial_id": cfg.trial_id,
            "treatment_arm": arms,
            "randomization_date": rand_dates,
            "age": ages.round(1),
            "sex": sexes,
            "race": races,
            "ecog": ecog,
            "tumor_burden": tumor_burden.round(2),
            **{k: v.round(2) for k, v in labs.items()},
        })

        # Store internal state for survival generation
        self._log_tumor = log_tumor
        self._sickness = sickness_factor

        return subjects

    def _generate_survival(self, subjects: pd.DataFrame) -> pd.DataFrame:
        """Generate OS times using Weibull model with covariate effects."""
        cfg = self.config
        n = len(subjects)
        cov_effects = cfg.get("covariate_effects")

        # Compute linear predictor (log-hazard)
        lp = np.zeros(n)
        lp += cov_effects["age"] * (subjects["age"].values - 62) / 10
        lp += cov_effects["ecog"] * subjects["ecog"].values
        lp += cov_effects["log_tumor"] * (self._log_tumor - 3.5)
        lp += cov_effects["log_ldh"] * (np.log(subjects["ldh"].values) - np.log(250))
        lp += cov_effects["albumin"] * (subjects["albumin"].values - 3.8)
        lp += cov_effects["hemoglobin"] * (subjects["hemoglobin"].values - 12.5)

        # Treatment effect
        is_treated = (subjects["treatment_arm"] == "treatment").values
        treatment_hr = cfg.get("survival", "treatment_hazard_ratio")
        lp[is_treated] += np.log(treatment_hr)

        # Weibull survival times
        shape = cfg.get("survival", "os_shape")
        median_months = cfg.get("survival", "os_baseline_median_months")
        baseline_scale = median_months * 30.44 / (np.log(2) ** (1 / shape))

        # Individual scale parameters
        individual_scale = baseline_scale * np.exp(-lp / shape)
        u = self.rng.uniform(0, 1, size=n)
        os_times = individual_scale * (-np.log(u)) ** (1 / shape)

        # Administrative censoring
        max_followup = cfg.get("survival", "max_followup_months") * 30.44
        censoring_rate = cfg.get("survival", "censoring_rate")
        random_censor = self.rng.exponential(max_followup / censoring_rate, size=n)
        censor_times = np.minimum(max_followup, random_censor)

        os_event = (os_times <= censor_times).astype(int)
        os_time = np.where(os_event, os_times, censor_times)

        return pd.DataFrame({
            "subject_id": subjects["subject_id"],
            "os_time": os_time.round(1),
            "os_event": os_event,
            "pfs_time": np.nan,  # filled later from tumor trajectory
            "pfs_event": np.nan,
        })

    def _generate_longitudinal(
        self, subjects: pd.DataFrame, endpoints: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate longitudinal tumor and lab measurements."""
        cfg = self.config
        visit_days = cfg.get("longitudinal", "visit_days")
        missing_rate = cfg.get("longitudinal", "missing_rate")
        lab_drift_std = cfg.get("longitudinal", "lab_drift_std")

        records = []
        for idx, row in subjects.iterrows():
            sid = row["subject_id"]
            is_treated = row["treatment_arm"] == "treatment"
            os_time = endpoints.loc[idx, "os_time"]
            baseline_tumor = row["tumor_burden"]

            # Tumor trajectory
            if is_treated:
                shrink_rate = cfg.get("tumor", "shrinkage_rate_treatment")
                will_regrow = self.rng.random() < cfg.get("tumor", "regrowth_probability")
                regrowth_day = self.rng.normal(
                    cfg.get("tumor", "regrowth_onset_days", "mean"),
                    cfg.get("tumor", "regrowth_onset_days", "std"),
                ) if will_regrow else np.inf
            else:
                growth_rate = cfg.get("tumor", "growth_rate_control")

            for day in visit_days:
                if day > os_time:
                    break

                # Skip visit with missing_rate probability (except baseline)
                if day > 0 and self.rng.random() < missing_rate:
                    continue

                # Tumor size
                if is_treated:
                    if day < regrowth_day:
                        tumor = baseline_tumor * np.exp(shrink_rate * day)
                    else:
                        nadir = baseline_tumor * np.exp(shrink_rate * regrowth_day)
                        tumor = nadir * np.exp(growth_rate * 0.8 * (day - regrowth_day))
                else:
                    tumor = baseline_tumor * np.exp(growth_rate * day)

                tumor += self.rng.normal(0, baseline_tumor * 0.05)
                tumor = max(0.1, tumor)

                # Lab values (random walk from baseline, clipped to plausible range)
                lab_values = {}
                for lab_name, lab_cfg in cfg.get("baseline_labs").items():
                    baseline_val = row[lab_name]
                    drift = self.rng.normal(0, baseline_val * lab_drift_std * np.sqrt(day / 30))
                    val = baseline_val + drift
                    nrange = lab_cfg.get("normal_range", [0.1, 1e6])
                    val = np.clip(val, nrange[0] * 0.25, nrange[1])
                    lab_values[lab_name] = max(0.1, val)

                # Adverse events (higher probability for treated, increasing over time)
                ae_prob = 0.05 + (0.1 if is_treated else 0.03) * (day / 365)
                ae_flag = int(self.rng.random() < ae_prob)

                records.append({
                    "subject_id": sid,
                    "days_since_randomization": day,
                    "visit_number": visit_days.index(day),
                    "tumor_size": round(tumor, 2),
                    "adverse_event_flag": ae_flag,
                    **{k: round(v, 2) for k, v in lab_values.items()},
                })

        return pd.DataFrame(records)

    def _derive_pfs(
        self, longitudinal: pd.DataFrame, endpoints: pd.DataFrame
    ) -> pd.DataFrame:
        """Derive PFS from tumor trajectory using RECIST-like 20% increase from nadir."""
        endpoints = endpoints.copy()

        for sid in endpoints["subject_id"].unique():
            subj_data = longitudinal[longitudinal["subject_id"] == sid].sort_values(
                "days_since_randomization"
            )
            if subj_data.empty:
                continue

            tumors = subj_data[["days_since_randomization", "tumor_size"]].values
            nadir = tumors[0, 1]
            progression_day = None

            for day, size in tumors[1:]:
                nadir = min(nadir, size)
                if size >= nadir * 1.20:  # RECIST-simplified: 20% increase from nadir
                    progression_day = day
                    break

            mask = endpoints["subject_id"] == sid
            os_time = endpoints.loc[mask, "os_time"].values[0]
            os_event = endpoints.loc[mask, "os_event"].values[0]

            if progression_day is not None:
                pfs_time = min(progression_day, os_time)
                pfs_event = 1
            elif os_event == 1:
                pfs_time = os_time
                pfs_event = 1
            else:
                # Censored: last tumor assessment
                last_assess = tumors[-1, 0] if len(tumors) > 0 else os_time
                pfs_time = min(last_assess, os_time)
                pfs_event = 0

            endpoints.loc[mask, "pfs_time"] = round(pfs_time, 1)
            endpoints.loc[mask, "pfs_event"] = int(pfs_event)

        return endpoints

    def _save_sdtm_csvs(
        self,
        subjects: pd.DataFrame,
        longitudinal: pd.DataFrame,
        endpoints: pd.DataFrame,
    ) -> None:
        """Save data as CDISC SDTM-formatted domain CSVs."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # DM domain
        dm = subjects.rename(columns={
            "subject_id": "USUBJID",
            "trial_id": "STUDYID",
            "treatment_arm": "ARMCD",
            "age": "AGE",
            "sex": "SEX",
            "race": "RACE",
        })
        dm["DOMAIN"] = "DM"
        dm["ARM"] = dm["ARMCD"].map(self.config.arms)
        dm["AGEU"] = "YEARS"
        dm.to_csv(self.output_dir / "dm.csv", index=False)

        # DS domain (disposition — contains death info)
        ds_records = []
        for _, row in endpoints.iterrows():
            ds_records.append({
                "STUDYID": self.config.trial_id,
                "DOMAIN": "DS",
                "USUBJID": row["subject_id"],
                "DSDECOD": "DEATH" if row["os_event"] == 1 else "COMPLETED",
                "DSSTDY": int(row["os_time"]),
            })
        pd.DataFrame(ds_records).to_csv(self.output_dir / "ds.csv", index=False)

        # TR domain (tumor results)
        tr = longitudinal[["subject_id", "days_since_randomization", "tumor_size"]].copy()
        tr = tr.rename(columns={
            "subject_id": "USUBJID",
            "days_since_randomization": "TRDY",
            "tumor_size": "TRSTRESN",
        })
        tr["STUDYID"] = self.config.trial_id
        tr["DOMAIN"] = "TR"
        tr["TRTESTCD"] = "TUMSTATE"
        tr["TRSTRESU"] = "cm"
        tr.to_csv(self.output_dir / "tr.csv", index=False)

        # LB domain (labs)
        lab_names = list(self.config.get("baseline_labs").keys())
        lb_records = []
        for _, row in longitudinal.iterrows():
            for lab in lab_names:
                lb_records.append({
                    "STUDYID": self.config.trial_id,
                    "DOMAIN": "LB",
                    "USUBJID": row["subject_id"],
                    "LBTESTCD": lab.upper(),
                    "LBTEST": lab.replace("_", " ").title(),
                    "LBSTRESN": row[lab],
                    "LBSTRESU": self.config.get("baseline_labs", lab, "unit"),
                    "LBDY": row["days_since_randomization"],
                })
        pd.DataFrame(lb_records).to_csv(self.output_dir / "lb.csv", index=False)

        # AE domain
        ae_rows = longitudinal[longitudinal["adverse_event_flag"] == 1]
        ae = ae_rows[["subject_id", "days_since_randomization"]].rename(columns={
            "subject_id": "USUBJID",
            "days_since_randomization": "AESTDY",
        })
        ae["STUDYID"] = self.config.trial_id
        ae["DOMAIN"] = "AE"
        ae["AETERM"] = "ADVERSE EVENT"
        ae.to_csv(self.output_dir / "ae.csv", index=False)
