"""
DecisionRouter: assigns each candidate pair to one of three tiers.

Tier 1 - AUTO_MERGE:     High-confidence duplicate. System merges automatically.
Tier 2 - HUMAN_REVIEW:   Ambiguous. System halts and diverts to human administrator.
Tier 3 - AUTO_DISTINCT:  High-confidence distinct individuals. No action taken.

Hard signals short-circuit the MARS model entirely:
- Exact unique ID match -> Tier 1 immediately
- Exact unique ID mismatch (both non-null) -> Tier 3 immediately
- Exact phone match + JW >= 0.7 -> Tier 1
- DOB exact match + JW >= 0.85 -> Tier 1
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Tier(str, Enum):
    AUTO_MERGE    = "auto_merge"
    HUMAN_REVIEW  = "human_review"
    AUTO_DISTINCT = "auto_distinct"


@dataclass
class RoutingResult:
    tier: Tier
    probability: Optional[float]
    conformal_set: frozenset
    reason: str
    features: dict
    mars_trace: Optional[dict] = None


class DecisionRouter:
    """
    Routes a candidate pair to a decision tier.

    Parameters
    ----------
    mars_scorer : MARSScorer
        The loaded MARS model.
    icp : ICPPredictor | None
        Optional conformal predictor. If None, uses probability thresholds only.
    high_prob_threshold : float
        Probability above which a pair is AUTO_MERGE (default 0.85).
    low_prob_threshold : float
        Probability below which a pair is AUTO_DISTINCT (default 0.15).
    """

    def __init__(
        self,
        mars_scorer,
        icp=None,
        high_prob_threshold: float = 0.85,
        low_prob_threshold: float = 0.15,
    ):
        self.scorer = mars_scorer
        self.icp = icp
        self.high_prob = high_prob_threshold
        self.low_prob = low_prob_threshold

    def route(self, features: dict, schema=None) -> RoutingResult:
        """
        Route a feature dict to a tier.

        Hard signal checks run first; MARS model runs only if no hard signal fires.
        """
        # Hard signal: exact ID
        exact_id_val = features.get("exact_id")
        if exact_id_val == 1.0:
            return RoutingResult(
                tier=Tier.AUTO_MERGE,
                probability=1.0,
                conformal_set=frozenset({1}),
                reason="hard_signal:exact_id_match",
                features=features,
            )
        if exact_id_val == 0.0:
            return RoutingResult(
                tier=Tier.AUTO_DISTINCT,
                probability=0.0,
                conformal_set=frozenset({0}),
                reason="hard_signal:exact_id_mismatch",
                features=features,
            )

        # Hard signal: phone + name
        phone_val = features.get("exact_phone")
        jw_full = features.get("jw_name_full", 0.0) or 0.0
        if phone_val == 1.0 and jw_full >= 0.7:
            return RoutingResult(
                tier=Tier.AUTO_MERGE,
                probability=0.99,
                conformal_set=frozenset({1}),
                reason="hard_signal:phone_match+name_similarity",
                features=features,
            )

        # Hard signal: DOB + name
        dob_val = features.get("dob_match")
        if dob_val == 1.0 and jw_full >= 0.85:
            return RoutingResult(
                tier=Tier.AUTO_MERGE,
                probability=0.97,
                conformal_set=frozenset({1}),
                reason="hard_signal:dob_match+name_similarity",
                features=features,
            )

        # MARS probabilistic scoring
        trace = self.scorer.score_with_trace(features)
        prob = trace["probability"]

        # ICP conformal set
        if self.icp:
            conf_set = self.icp.predict_set(prob)
        else:
            if prob >= self.high_prob:
                conf_set = frozenset({1})
            elif prob <= self.low_prob:
                conf_set = frozenset({0})
            else:
                conf_set = frozenset({0, 1})

        # Tier assignment
        if conf_set == frozenset({1}):
            tier = Tier.AUTO_MERGE
            reason = f"mars:high_confidence_duplicate(p={prob:.3f})"
        elif conf_set == frozenset({0}):
            tier = Tier.AUTO_DISTINCT
            reason = f"mars:high_confidence_distinct(p={prob:.3f})"
        else:
            tier = Tier.HUMAN_REVIEW
            reason = f"mars:ambiguous_zone(p={prob:.3f},jw={jw_full:.3f})"

        return RoutingResult(
            tier=tier,
            probability=prob,
            conformal_set=conf_set,
            reason=reason,
            features=features,
            mars_trace=trace,
        )
