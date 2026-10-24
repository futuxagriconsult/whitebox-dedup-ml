"""
Tests for DecisionRouter: tier assignment and hard signal logic.
"""
import pytest
from whitebox_dedup.mars_scorer import MARSScorer
from whitebox_dedup.conformal import ICPPredictor
from whitebox_dedup.router import DecisionRouter, Tier, RoutingResult


@pytest.fixture
def scorer():
    return MARSScorer.load_pretrained("text_only_v1")


@pytest.fixture
def icp():
    return ICPPredictor.load_pretrained("text_only_v1")


@pytest.fixture
def router(scorer, icp):
    return DecisionRouter(scorer, icp)


@pytest.fixture
def router_no_icp(scorer):
    return DecisionRouter(scorer, icp=None)


# ---- Hard signal: exact ID match ----

def test_exact_id_match_routes_to_auto_merge(router):
    feats = {"exact_id": 1.0, "jw_name_full": 0.5}
    result = router.route(feats)
    assert result.tier == Tier.AUTO_MERGE


def test_exact_id_match_reason_contains_hard_signal(router):
    feats = {"exact_id": 1.0}
    result = router.route(feats)
    assert "hard_signal" in result.reason
    assert "exact_id_match" in result.reason


def test_exact_id_match_probability_is_one(router):
    feats = {"exact_id": 1.0}
    result = router.route(feats)
    assert result.probability == 1.0


def test_exact_id_match_overrides_low_jw(router):
    """Exact ID match must yield AUTO_MERGE even when name similarity is very low."""
    feats = {"exact_id": 1.0, "jw_name_full": 0.1}
    result = router.route(feats)
    assert result.tier == Tier.AUTO_MERGE


# ---- Hard signal: exact ID mismatch ----

def test_exact_id_mismatch_routes_to_auto_distinct(router):
    feats = {"exact_id": 0.0, "jw_name_full": 0.99}
    result = router.route(feats)
    assert result.tier == Tier.AUTO_DISTINCT


def test_exact_id_mismatch_reason_contains_hard_signal(router):
    feats = {"exact_id": 0.0}
    result = router.route(feats)
    assert "hard_signal" in result.reason
    assert "exact_id_mismatch" in result.reason


def test_exact_id_mismatch_overrides_high_jw(router):
    """Exact ID mismatch must yield AUTO_DISTINCT even when name similarity is very high."""
    feats = {"exact_id": 0.0, "jw_name_full": 0.99}
    result = router.route(feats)
    assert result.tier == Tier.AUTO_DISTINCT


# ---- Hard signal: phone match + name similarity ----

def test_phone_match_high_jw_routes_to_auto_merge(router):
    feats = {"exact_phone": 1.0, "jw_name_full": 0.85}
    result = router.route(feats)
    assert result.tier == Tier.AUTO_MERGE


def test_phone_match_low_jw_does_not_hard_signal(router):
    """Phone match alone (low JW) should not trigger hard signal."""
    feats = {"exact_phone": 1.0, "jw_name_full": 0.4}
    result = router.route(feats)
    # No hard signal fires; falls through to MARS model
    assert "hard_signal:phone" not in result.reason


# ---- Hard signal: DOB match + name similarity ----

def test_dob_match_high_jw_routes_to_auto_merge(router):
    feats = {"dob_match": 1.0, "jw_name_full": 0.90}
    result = router.route(feats)
    assert result.tier == Tier.AUTO_MERGE


def test_dob_match_low_jw_does_not_hard_signal(router):
    feats = {"dob_match": 1.0, "jw_name_full": 0.5}
    result = router.route(feats)
    assert "hard_signal:dob" not in result.reason


# ---- MARS probabilistic routing ----

def test_high_jw_routes_to_auto_merge(router):
    """JW = 1.0 must yield a high probability and AUTO_MERGE."""
    feats = {"jw_name_full": 1.0}
    result = router.route(feats)
    assert result.tier == Tier.AUTO_MERGE


def test_low_jw_routes_to_auto_distinct(router_no_icp):
    """Very low JW must yield AUTO_DISTINCT via probability threshold."""
    feats = {"jw_name_full": 0.1}
    result = router_no_icp.route(feats)
    assert result.tier == Tier.AUTO_DISTINCT


def test_ambiguous_jw_routes_to_human_review(router):
    """
    The ICP calibration thresholds for text_only_v1 define an ambiguous zone.
    A probability of 0.5 sits in that zone: nc_1=0.5 > threshold_duplicate=0.23
    and nc_0=0.5 > threshold_distinct=0.18, so the conformal set is {0,1}.
    We feed a feature dict with only the jw key absent (score defaults to intercept-only
    which gives near-zero probability, routing to AUTO_DISTINCT).
    Instead we construct a synthetic router with wider thresholds to test the path.
    """
    from whitebox_dedup.conformal import ICPCalibration
    # Calibration where both thresholds are 0.6: anything between 0.4 and 0.6 is ambiguous
    wide_cal = ICPCalibration(alpha=0.05, threshold_duplicate=0.6, threshold_distinct=0.6)
    from whitebox_dedup.conformal import ICPPredictor
    wide_icp = ICPPredictor(wide_cal)
    scorer = MARSScorer.load_pretrained("text_only_v1")
    wide_router = DecisionRouter(scorer, wide_icp)
    # Probability 0.5: nc_1=0.5 <= 0.6 so 1 included; nc_0=0.5 <= 0.6 so 0 included -> {0,1}
    # Use a feature dict that is not handled by hard signals
    # Synthetic: pass only a custom feature that MARS scorer ignores (returns intercept prob ~ 0)
    # Instead, directly verify the ICP path by providing a mid probability
    # Patch: use a dummy scorer that returns 0.5 always
    class DummyScorer:
        model_name = "text_only_v1"
        def score_with_trace(self, features):
            return {"probability": 0.5, "logit": 0.0, "intercept": 0.0, "hinge_contributions": []}
    dummy_router = DecisionRouter(DummyScorer(), wide_icp)
    result = dummy_router.route({"jw_name_full": 0.65})
    assert result.tier == Tier.HUMAN_REVIEW


# ---- RoutingResult structure ----

def test_routing_result_has_all_fields(router):
    feats = {"jw_name_full": 0.9}
    result = router.route(feats)
    assert isinstance(result, RoutingResult)
    assert result.tier is not None
    assert result.reason is not None
    assert result.features is not None
    assert result.conformal_set is not None


def test_routing_result_mars_trace_present_for_probabilistic(router):
    """MARS trace must be present when the MARS model is invoked."""
    feats = {"jw_name_full": 0.9}
    result = router.route(feats)
    assert result.mars_trace is not None


def test_routing_result_no_mars_trace_for_hard_signal(router):
    """Hard signal routes should not have a mars_trace (skipped MARS model)."""
    feats = {"exact_id": 1.0}
    result = router.route(feats)
    assert result.mars_trace is None
