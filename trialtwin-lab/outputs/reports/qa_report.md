# Clinical Data QA Report

**8/8 checks passed**

| Check | Status | Message |
|-------|--------|---------|
| No Negative Survival Times | PASS | All survival times are non-negative |
| No Post-Death Measurements | PASS | No post-death measurements found |
| Plausible Lab Ranges | PASS | All lab values within plausible ranges |
| Monotonic Visit Times | PASS | All visit times are monotonically increasing |
| No Future Data Leakage | PASS | Longitudinal data extends beyond landmark day 60 (feature engineering must filter to <= 60) |
| Consistent Censoring | PASS | All censoring is consistent |
| Complete Demographics | PASS | All required demographics are complete |
| Balanced Randomization | PASS | Arm distribution: {'control': 157, 'treatment': 143} (balanced) |

## Details

**No Negative Survival Times**: OS negative: 0, PFS negative: 0

**No Future Data Leakage**: Check is informational — leakage prevention enforced in feature builder