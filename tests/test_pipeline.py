"""
Tests for DeduplicationPipeline: end-to-end deduplication, model selection, and scoring.
"""
from pathlib import Path
import pytest
import pandas as pd
from whitebox_dedup.pipeline import DeduplicationPipeline, DeduplicationResult
from whitebox_dedup.schema import EntitySchema, EntityField, FieldRole
from whitebox_dedup.router import RoutingResult, Tier


FIXTURES = Path(__file__).parent / "fixtures"
FARMER_CSV = FIXTURES / "registry_farmer_200.csv"
NIN_CSV = FIXTURES / "registry_nin_300.csv"


# ---- DeduplicationPipeline.from_csv ----

def test_from_csv_loads_farmer_registry():
    pipeline = DeduplicationPipeline.from_csv(FARMER_CSV)
    assert pipeline.df is not None
    assert len(pipeline.df) == 200


def test_from_csv_infers_schema_with_name_role():
    pipeline = DeduplicationPipeline.from_csv(FARMER_CSV)
    assert pipeline.schema.has_role(FieldRole.NAME)


def test_from_csv_selects_model():
    pipeline = DeduplicationPipeline.from_csv(FARMER_CSV)
    assert pipeline.scorer is not None
    assert pipeline.scorer.model_name is not None


# ---- DeduplicationPipeline.run on farmer registry ----

def test_pipeline_run_farmer_returns_result():
    pipeline = DeduplicationPipeline.from_csv(FARMER_CSV)
    result = pipeline.run()
    assert isinstance(result, DeduplicationResult)


def test_pipeline_run_farmer_finds_known_duplicates():
    """
    The farmer registry contains 7 known duplicate pairs.
    At least some of them should be detected as AUTO_MERGE or HUMAN_REVIEW.
    """
    pipeline = DeduplicationPipeline.from_csv(FARMER_CSV)
    result = pipeline.run()
    detected = len(result.tier1_auto_merge) + len(result.tier2_human_review)
    assert detected > 0


def test_pipeline_run_farmer_total_records_correct():
    pipeline = DeduplicationPipeline.from_csv(FARMER_CSV)
    result = pipeline.run()
    assert result.total_records == 200


def test_pipeline_run_candidate_pairs_positive():
    pipeline = DeduplicationPipeline.from_csv(FARMER_CSV)
    result = pipeline.run()
    assert result.total_candidate_pairs > 0


# ---- DeduplicationPipeline on NIN registry (exact ID hard signals) ----

def test_pipeline_run_nin_detects_exact_nin_matches():
    """
    The NIN registry has 5 NINs that appear exactly twice.
    All 5 should be detected as Tier 1 AUTO_MERGE via the exact_id hard signal.
    """
    pipeline = DeduplicationPipeline.from_csv(NIN_CSV)
    result = pipeline.run()
    exact_id_merges = [
        p for p in result.tier1_auto_merge
        if "exact_id_match" in p.get("reason", "")
    ]
    assert len(exact_id_merges) >= 5


def test_pipeline_run_nin_has_auto_merge():
    pipeline = DeduplicationPipeline.from_csv(NIN_CSV)
    result = pipeline.run()
    assert len(result.tier1_auto_merge) >= 5


# ---- score_pair ----

def test_score_pair_returns_routing_result():
    pipeline = DeduplicationPipeline.from_csv(FARMER_CSV)
    a = {"full_name": "Aminu Bello", "phone_number": "08012345678",
         "state": "Kano", "lga": "Kano Municipal",
         "latitude": 12.0, "longitude": 8.5}
    b = {"full_name": "Aminu Bello", "phone_number": "08012345678",
         "state": "Kano", "lga": "Kano Municipal",
         "latitude": 12.0, "longitude": 8.5}
    result = pipeline.score_pair(a, b)
    assert isinstance(result, RoutingResult)


def test_score_pair_identical_records_auto_merge():
    pipeline = DeduplicationPipeline.from_csv(FARMER_CSV)
    record = {"full_name": "Fatima Mohammed", "phone_number": "08056789012",
              "state": "Kaduna", "lga": "Kaduna North",
              "latitude": 10.5, "longitude": 7.4}
    result = pipeline.score_pair(record, record)
    assert result.tier == Tier.AUTO_MERGE


def test_score_pair_very_different_records_auto_distinct():
    pipeline = DeduplicationPipeline.from_csv(FARMER_CSV)
    a = {"full_name": "Aminu Bello", "phone_number": "08012345678",
         "state": "Kano", "latitude": 12.0, "longitude": 8.5}
    b = {"full_name": "Zainab Waziri", "phone_number": "09098765432",
         "state": "Sokoto", "latitude": 13.0, "longitude": 5.0}
    result = pipeline.score_pair(a, b)
    assert result.tier in (Tier.AUTO_DISTINCT, Tier.HUMAN_REVIEW)


# ---- Model selection ----

def test_model_selection_without_gps_uses_text_model():
    """Schema without GPS should NOT select text_phone_geo_v1."""
    schema = EntitySchema(fields={
        "full_name": EntityField("full_name", FieldRole.NAME),
        "phone": EntityField("phone", FieldRole.PHONE),
    })
    df = pd.DataFrame([{"full_name": "Aminu Bello", "phone": "08012345678"}])
    pipeline = DeduplicationPipeline.from_schema(schema, df)
    assert "geo" not in pipeline.scorer.model_name


def test_model_selection_name_only_selects_text_only():
    """Schema with name only must select text_only_v1."""
    schema = EntitySchema(fields={
        "full_name": EntityField("full_name", FieldRole.NAME),
    })
    df = pd.DataFrame([{"full_name": "Aminu Bello"}])
    pipeline = DeduplicationPipeline.from_schema(schema, df)
    assert pipeline.scorer.model_name == "text_only_v1"


# ---- DeduplicationResult.summary ----

def test_summary_keys():
    result = DeduplicationResult(total_records=100, total_candidate_pairs=50)
    summary = result.summary()
    assert "total_records" in summary
    assert "candidate_pairs_evaluated" in summary
    assert "tier1_auto_merge" in summary
    assert "tier2_human_review" in summary
    assert "tier3_auto_distinct" in summary


def test_summary_counts_match_lists():
    result = DeduplicationResult(
        total_records=200,
        total_candidate_pairs=30,
        tier1_auto_merge=[{"a": 1}, {"a": 2}],
        tier2_human_review=[{"b": 1}],
        tier3_auto_distinct=[{"c": 1}, {"c": 2}, {"c": 3}],
    )
    s = result.summary()
    assert s["tier1_auto_merge"] == 2
    assert s["tier2_human_review"] == 1
    assert s["tier3_auto_distinct"] == 3


def test_repr_contains_record_count():
    result = DeduplicationResult(total_records=200, total_candidate_pairs=30)
    r = repr(result)
    assert "200" in r


# ---- from_schema ----

def test_from_schema_with_dataframe():
    schema = EntitySchema(fields={
        "full_name": EntityField("full_name", FieldRole.NAME),
    })
    df = pd.DataFrame([
        {"full_name": "Aminu Bello"},
        {"full_name": "Aminu Bello"},
        {"full_name": "Zainab Waziri"},
    ])
    pipeline = DeduplicationPipeline.from_schema(schema, df)
    result = pipeline.run()
    assert result.total_records == 3
    assert result.total_candidate_pairs >= 1
