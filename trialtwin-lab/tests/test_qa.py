"""Tests for QA checks: all 8 should pass on synthetic data."""

from trialtwin.qa.checks import run_all_checks, ALL_CHECKS


def test_all_checks_pass(harmonized, nsclc_config):
    results = run_all_checks(harmonized, nsclc_config)
    failed = [r for r in results if not r.passed]
    assert len(failed) == 0, f"Failed checks: {[r.name for r in failed]}"


def test_eight_checks_registered():
    assert len(ALL_CHECKS) == 8


def test_each_check_returns_result(harmonized, nsclc_config):
    for check in ALL_CHECKS:
        result = check.run(harmonized, nsclc_config)
        assert hasattr(result, "name")
        assert hasattr(result, "passed")
        assert hasattr(result, "message")
        assert result.passed in (True, False)
