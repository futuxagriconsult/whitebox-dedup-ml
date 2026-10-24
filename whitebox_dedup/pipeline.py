"""
DeduplicationPipeline: end-to-end entity deduplication.

1. Generates candidate pairs using blocking
2. Extracts features for each pair
3. Routes each pair to AUTO_MERGE, HUMAN_REVIEW, or AUTO_DISTINCT
4. Returns structured results
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
from .schema import EntitySchema
from .blocking import generate_candidate_pairs
from .features import extract_all
from .mars_scorer import MARSScorer
from .conformal import ICPPredictor
from .router import DecisionRouter, RoutingResult, Tier


@dataclass
class DeduplicationResult:
    total_records: int
    total_candidate_pairs: int
    tier1_auto_merge: list[dict] = field(default_factory=list)
    tier2_human_review: list[dict] = field(default_factory=list)
    tier3_auto_distinct: list[dict] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "total_records": self.total_records,
            "candidate_pairs_evaluated": self.total_candidate_pairs,
            "tier1_auto_merge": len(self.tier1_auto_merge),
            "tier2_human_review": len(self.tier2_human_review),
            "tier3_auto_distinct": len(self.tier3_auto_distinct),
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"DeduplicationResult("
            f"records={s['total_records']}, "
            f"candidates={s['candidate_pairs_evaluated']}, "
            f"auto_merge={s['tier1_auto_merge']}, "
            f"human_review={s['tier2_human_review']}, "
            f"auto_distinct={s['tier3_auto_distinct']})"
        )


class DeduplicationPipeline:
    """
    Full deduplication pipeline.

    Examples
    --------
    Auto-infer schema and use default pre-trained model:
    >>> pipeline = DeduplicationPipeline.from_csv("farmers.csv")
    >>> results = pipeline.run()
    >>> print(results)

    Explicit schema:
    >>> schema = EntitySchema.infer("nin_records.csv")
    >>> pipeline = DeduplicationPipeline.from_schema(schema, "nin_records.csv")
    >>> results = pipeline.run()

    Score a single pair without running full dedup:
    >>> result = pipeline.score_pair(record_a_dict, record_b_dict)
    """

    _MODEL_SELECTION_ORDER = [
        ("text_phone_geo_v1", {"jw_name_full", "exact_phone", "haversine_m"}),
        ("text_phone_v1",     {"jw_name_full", "exact_phone"}),
        ("text_only_v1",      {"jw_name_full"}),
    ]

    def __init__(
        self,
        df: pd.DataFrame,
        schema: EntitySchema,
        scorer: MARSScorer,
        icp: ICPPredictor | None = None,
        geo_radius_m: float = 50.0,
    ):
        self.df = df
        self.schema = schema
        self.scorer = scorer
        self.icp = icp
        self.geo_radius_m = geo_radius_m
        self.router = DecisionRouter(scorer, icp)

    @classmethod
    def from_csv(cls, path: str | Path, schema: EntitySchema | None = None) -> "DeduplicationPipeline":
        df = pd.read_csv(path)
        schema = schema or EntitySchema.infer(df)
        scorer = cls._load_best_model(schema)
        try:
            icp = ICPPredictor.load_pretrained(scorer.model_name)
        except FileNotFoundError:
            icp = None
        return cls(df=df, schema=schema, scorer=scorer, icp=icp)

    @classmethod
    def from_schema(cls, schema: EntitySchema, source: str | Path | pd.DataFrame) -> "DeduplicationPipeline":
        if isinstance(source, (str, Path)):
            df = pd.read_csv(source)
        else:
            df = source.copy()
        scorer = cls._load_best_model(schema)
        try:
            icp = ICPPredictor.load_pretrained(scorer.model_name)
        except FileNotFoundError:
            icp = None
        return cls(df=df, schema=schema, scorer=scorer, icp=icp)

    @classmethod
    def _load_best_model(cls, schema: EntitySchema) -> MARSScorer:
        active = schema.active_features()
        for model_name, required_feats in cls._MODEL_SELECTION_ORDER:
            if required_feats.issubset(active):
                try:
                    return MARSScorer.load_pretrained(model_name)
                except FileNotFoundError:
                    continue
        # Fallback: text_only
        return MARSScorer.load_pretrained("text_only_v1")

    def run(self) -> DeduplicationResult:
        """Run full deduplication pipeline."""
        candidate_pairs = generate_candidate_pairs(self.df, self.schema, self.geo_radius_m)
        records = self.df.to_dict(orient="records")

        result = DeduplicationResult(
            total_records=len(self.df),
            total_candidate_pairs=len(candidate_pairs),
        )

        for idx_a, idx_b in candidate_pairs:
            row_a = records[self.df.index.get_loc(idx_a)]
            row_b = records[self.df.index.get_loc(idx_b)]
            feats = extract_all(row_a, row_b, self.schema)
            routing = self.router.route(feats, self.schema)

            pair_record = {
                "index_a": idx_a,
                "index_b": idx_b,
                "tier": routing.tier.value,
                "probability": routing.probability,
                "reason": routing.reason,
                "features": {k: v for k, v in feats.items() if v is not None},
            }

            if routing.tier == Tier.AUTO_MERGE:
                result.tier1_auto_merge.append(pair_record)
            elif routing.tier == Tier.HUMAN_REVIEW:
                result.tier2_human_review.append(pair_record)
            else:
                result.tier3_auto_distinct.append(pair_record)

        return result

    def score_pair(self, row_a: dict, row_b: dict) -> RoutingResult:
        """Score a single pair without running full blocking."""
        feats = extract_all(row_a, row_b, self.schema)
        return self.router.route(feats, self.schema)
