"""
Tests for Inductive Conformal Prediction module.
"""
import pytest
from whitebox_dedup.conformal import ICPCalibration, ICPPredictor


@pytest.fixture
def icp():
    return ICPPredictor.load_pretrained("text_only_v1")


# ---- predict_set: clear duplicate ----

def test_predict_set_high_probability_returns_duplicate(icp):
    """Probability 0.95 should include only class 1 (duplicate)."""
    pred_set = icp.predict_set(0.95)
    assert 1 in pred_set
    # The nonconformity for class 1 is 1-0.95 = 0.05 <= threshold_duplicate=0.23
    # The nonconformity for class 0 is 0.95 > threshold_distinct=0.18
    assert 0 not in pred_set


def test_predict_set_high_probability_is_singleton(icp):
    pred_set = icp.predict_set(0.95)
    assert pred_set == frozenset({1})


# ---- predict_set: clear distinct ----

def test_predict_set_low_probability_returns_distinct(icp):
    """Probability 0.05 should include only class 0 (distinct)."""
    pred_set = icp.predict_set(0.05)
    assert 0 in pred_set
    # nc_0 = 0.05 <= threshold_distinct=0.18
    # nc_1 = 0.95 > threshold_duplicate=0.23
    assert 1 not in pred_set


def test_predict_set_low_probability_is_singleton(icp):
    pred_set = icp.predict_set(0.05)
    assert pred_set == frozenset({0})


# ---- predict_set: ambiguous zone ----

def test_predict_set_ambiguous_probability_returns_both(icp):
    """Probability 0.5 should return {0, 1} (circuit breaker)."""
    pred_set = icp.predict_set(0.5)
    # nc_1 = 0.5 > threshold_duplicate=0.23 so 1 not included
    # nc_0 = 0.5 > threshold_distinct=0.18 so 0 not included
    # fallback: return {0, 1}
    assert pred_set == frozenset({0, 1})


def test_predict_set_always_non_empty(icp):
    """Prediction set must never be empty for any probability."""
    for p in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        pred_set = icp.predict_set(p)
        assert len(pred_set) > 0, f"Empty prediction set at p={p}"


# ---- is_ambiguous ----

def test_is_ambiguous_returns_true_for_midrange(icp):
    assert icp.is_ambiguous(0.5) is True


def test_is_ambiguous_returns_false_for_high_probability(icp):
    assert icp.is_ambiguous(0.95) is False


def test_is_ambiguous_returns_false_for_low_probability(icp):
    assert icp.is_ambiguous(0.05) is False


# ---- coverage_zone ----

def test_coverage_zone_below_491(icp):
    assert icp.coverage_zone(0.3) == "high_coverage"


def test_coverage_zone_ambiguous_range(icp):
    assert icp.coverage_zone(0.65) == "ambiguous_zone"


def test_coverage_zone_above_898(icp):
    assert icp.coverage_zone(0.95) == "high_coverage"


def test_coverage_zone_boundary_491(icp):
    """Exactly at 0.491 enters the ambiguous zone."""
    assert icp.coverage_zone(0.491) == "ambiguous_zone"


def test_coverage_zone_boundary_898(icp):
    """Exactly at 0.898 returns to high_coverage."""
    assert icp.coverage_zone(0.898) == "high_coverage"


# ---- load_pretrained error handling ----

def test_load_pretrained_missing_raises():
    with pytest.raises(FileNotFoundError):
        ICPPredictor.load_pretrained("nonexistent_calibration_xyz")


# ---- ICPCalibration dataclass ----

def test_icp_calibration_attributes():
    cal = ICPCalibration(alpha=0.05, threshold_duplicate=0.23, threshold_distinct=0.18, model_name="test")
    assert cal.alpha == 0.05
    assert cal.threshold_duplicate == 0.23
    assert cal.threshold_distinct == 0.18
    assert cal.model_name == "test"
