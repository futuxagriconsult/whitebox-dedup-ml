"""
Tests for MARSScorer: model loading, scoring, tracing, and serialisation.
"""
import json
import math
import tempfile
from pathlib import Path
import pytest
from whitebox_dedup.mars_scorer import MARSScorer, HingeFunction


MODELS_DIR = Path(__file__).parent.parent / "models"


@pytest.fixture
def text_only_scorer():
    return MARSScorer.load_pretrained("text_only_v1")


# ---- Probability boundary tests ----

def test_score_identical_names_high_probability(text_only_scorer):
    """JW = 1.0 should yield probability near 1.0."""
    prob = text_only_scorer.score({"jw_name_full": 1.0})
    assert prob > 0.98


def test_score_zero_similarity_low_probability(text_only_scorer):
    """JW = 0.0 should yield probability near 0.0."""
    prob = text_only_scorer.score({"jw_name_full": 0.0})
    assert prob < 0.01


def test_score_intermediate_jw(text_only_scorer):
    """JW = 0.75 should yield a probability between 0 and 1."""
    prob = text_only_scorer.score({"jw_name_full": 0.75})
    assert 0.0 < prob < 1.0


def test_score_above_threshold_high(text_only_scorer):
    """JW = 0.95 (well above 0.898 knot) should yield high probability."""
    prob = text_only_scorer.score({"jw_name_full": 0.95})
    assert prob > 0.8


def test_score_below_threshold_low(text_only_scorer):
    """JW = 0.3 (below 0.559 knot) should yield very low probability."""
    prob = text_only_scorer.score({"jw_name_full": 0.3})
    assert prob < 0.05


def test_score_missing_feature_returns_valid_probability(text_only_scorer):
    """Missing feature should not raise an error; returns intercept-based probability."""
    prob = text_only_scorer.score({})
    assert 0.0 <= prob <= 1.0


def test_score_none_feature_treated_as_missing(text_only_scorer):
    """None feature value should be treated as missing (0 contribution)."""
    prob_none = text_only_scorer.score({"jw_name_full": None})
    prob_empty = text_only_scorer.score({})
    assert prob_none == pytest.approx(prob_empty, abs=1e-9)


# ---- score_with_trace ----

def test_score_with_trace_structure(text_only_scorer):
    """Trace dict must contain all required keys."""
    trace = text_only_scorer.score_with_trace({"jw_name_full": 0.9})
    assert "probability" in trace
    assert "logit" in trace
    assert "intercept" in trace
    assert "hinge_contributions" in trace


def test_score_with_trace_hinge_count(text_only_scorer):
    """text_only_v1 has exactly 4 hinges."""
    trace = text_only_scorer.score_with_trace({"jw_name_full": 0.9})
    assert len(trace["hinge_contributions"]) == 4


def test_score_with_trace_contributions_sum(text_only_scorer):
    """Sum of hinge outputs + intercept must equal the logit."""
    jw = 0.85
    trace = text_only_scorer.score_with_trace({"jw_name_full": jw})
    total = trace["intercept"] + sum(c["hinge_output"] for c in trace["hinge_contributions"])
    assert total == pytest.approx(trace["logit"], abs=1e-3)


def test_score_with_trace_probability_consistent(text_only_scorer):
    """Probability from trace must match direct score()."""
    jw = 0.92
    direct = text_only_scorer.score({"jw_name_full": jw})
    trace = text_only_scorer.score_with_trace({"jw_name_full": jw})
    assert direct == pytest.approx(trace["probability"], abs=1e-5)


# ---- equation_text ----

def test_equation_text_contains_intercept(text_only_scorer):
    text = text_only_scorer.equation_text()
    assert "-102.231" in text


def test_equation_text_contains_feature_name(text_only_scorer):
    text = text_only_scorer.equation_text()
    assert "jw_name_full" in text


def test_equation_text_contains_sigmoid_formula(text_only_scorer):
    text = text_only_scorer.equation_text()
    assert "1 + exp(-logit)" in text


# ---- save / load roundtrip ----

def test_save_load_roundtrip_identical_scores(text_only_scorer):
    """Saving then loading a model must produce identical scores."""
    jw_values = [0.0, 0.3, 0.56, 0.75, 0.85, 0.92, 1.0]
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    text_only_scorer.save(tmp_path)
    loaded = MARSScorer.load(tmp_path)

    for jw in jw_values:
        original = text_only_scorer.score({"jw_name_full": jw})
        restored = loaded.score({"jw_name_full": jw})
        assert original == pytest.approx(restored, abs=1e-9), f"Mismatch at JW={jw}"


def test_save_load_roundtrip_hinge_count(text_only_scorer):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name
    text_only_scorer.save(tmp_path)
    loaded = MARSScorer.load(tmp_path)
    assert len(loaded.hinges) == len(text_only_scorer.hinges)


def test_save_load_roundtrip_model_name(text_only_scorer):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name
    text_only_scorer.save(tmp_path)
    loaded = MARSScorer.load(tmp_path)
    assert loaded.model_name == text_only_scorer.model_name


# ---- load_pretrained error handling ----

def test_load_pretrained_missing_model_raises():
    with pytest.raises(FileNotFoundError):
        MARSScorer.load_pretrained("nonexistent_model_xyz")


# ---- HingeFunction direct tests ----

def test_hinge_forward_evaluates_zero_below_knot():
    h = HingeFunction(feature="x", coefficient=2.0, knot=0.5, reflected=False)
    assert h.evaluate(0.3) == 0.0


def test_hinge_forward_evaluates_correctly_above_knot():
    h = HingeFunction(feature="x", coefficient=2.0, knot=0.5, reflected=False)
    assert h.evaluate(0.8) == pytest.approx(2.0 * (0.8 - 0.5), abs=1e-9)


def test_hinge_reflected_evaluates_zero_above_knot():
    h = HingeFunction(feature="x", coefficient=3.0, knot=0.7, reflected=True)
    assert h.evaluate(0.9) == 0.0


def test_hinge_reflected_evaluates_correctly_below_knot():
    h = HingeFunction(feature="x", coefficient=3.0, knot=0.7, reflected=True)
    assert h.evaluate(0.4) == pytest.approx(3.0 * (0.7 - 0.4), abs=1e-9)
