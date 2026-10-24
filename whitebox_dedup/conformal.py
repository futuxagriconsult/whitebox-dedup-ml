"""
Inductive Conformal Prediction (ICP) for deduplication.

ICP wraps the MARS scorer to produce a prediction SET {0}, {1}, or {0,1}
at a user-specified coverage level (e.g., 95%).

When the prediction set is {0,1} (ambiguous), the circuit breaker fires:
the system cannot make an automated decision and diverts to human review.

Coverage stratified by Jaro-Winkler similarity shows that coverage collapses
in the ambiguous zone (JW >= 0.491), establishing an automated statutory trigger.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ICPCalibration:
    """Pre-computed nonconformity score thresholds."""
    alpha: float                    # significance level (e.g., 0.05 for 95% coverage)
    threshold_duplicate: float      # nonconformity threshold for class 1 (duplicate)
    threshold_distinct: float       # nonconformity threshold for class 0 (distinct)
    model_name: str = ""

    @classmethod
    def load(cls, path: str | Path) -> "ICPCalibration":
        with open(path) as f:
            data = json.load(f)
        return cls(
            alpha=data["alpha"],
            threshold_duplicate=data["threshold_duplicate"],
            threshold_distinct=data["threshold_distinct"],
            model_name=data.get("model_name", ""),
        )


class ICPPredictor:
    """
    Produces conformal prediction sets given a MARS probability score.

    A prediction set of {1} means "definitely duplicate at the stated coverage level".
    A prediction set of {0} means "definitely distinct".
    A prediction set of {0, 1} means "ambiguous: circuit breaker fires".
    """

    def __init__(self, calibration: ICPCalibration):
        self.cal = calibration

    @classmethod
    def load_pretrained(cls, model_name: str) -> "ICPPredictor":
        cal_dir = Path(__file__).parent.parent / "models" / "calibration"
        path = cal_dir / f"{model_name}_icp.json"
        if not path.exists():
            raise FileNotFoundError(f"ICP calibration not found: {path}")
        return cls(ICPCalibration.load(path))

    def predict_set(self, probability: float) -> frozenset[int]:
        """
        Return the conformal prediction set for a given duplicate probability.

        The nonconformity score for class 1 is (1 - p), for class 0 is p.
        A label y is included in the prediction set if its nonconformity
        score <= the calibration threshold.
        """
        nc_1 = 1.0 - probability     # nonconformity for "duplicate"
        nc_0 = probability           # nonconformity for "distinct"

        include_1 = nc_1 <= self.cal.threshold_duplicate
        include_0 = nc_0 <= self.cal.threshold_distinct

        result = set()
        if include_1:
            result.add(1)
        if include_0:
            result.add(0)
        if not result:
            result = {0, 1}   # always return non-empty set
        return frozenset(result)

    def is_ambiguous(self, probability: float) -> bool:
        return self.predict_set(probability) == frozenset({0, 1})

    def coverage_zone(self, jw_score: float) -> str:
        """
        Classify coverage zone from JW similarity.
        Based on empirical finding: coverage collapses to 77.93% when JW >= 0.491.
        """
        if jw_score < 0.491:
            return "high_coverage"       # >= 95% empirical coverage
        elif jw_score >= 0.898:
            return "high_coverage"       # very high similarity -> clear duplicate
        else:
            return "ambiguous_zone"      # coverage 77.93% -> circuit breaker
