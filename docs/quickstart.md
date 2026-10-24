# whitebox-dedup Quickstart Guide

`whitebox-dedup` provides interpretable entity deduplication using mathematically auditable Multivariate Adaptive Regression Splines (MARS) and Inductive Conformal Prediction (ICP).

## Authors
- **Daniel Olagunju** (`d.olagunju@futuxconsult.com`)
- **Temitope Olarewaju** (`t.olarewaju@futuxconsult.com`)
- **Affiliation:** FutuX Agri-consult Limited

---

## 1. Installation

```bash
pip install whitebox-dedup
```

---

## 2. Python API

```python
from whitebox_dedup import DeduplicationPipeline, EntitySchema

# 1. Infer schema and run full deduplication pipeline
pipeline = DeduplicationPipeline.from_csv("farmers.csv")
results = pipeline.run()

print(f"Total evaluated records: {results.total_records}")
print(f"Tier 1 Auto-Merge (Confirmed Duplicates):  {len(results.tier1_auto_merge)}")
print(f"Tier 2 Human Review (Borderline Ambiguous): {len(results.tier2_human_review)}")
print(f"Tier 3 Auto-Distinct (Confirmed Distinct): {len(results.tier3_auto_distinct)}")

# 2. Score a single candidate pair interactively
pair_result = pipeline.score_pair(
    record_a={"full_name": "Aminu Bello", "phone": "+2348012345678"},
    record_b={"full_name": "Aminu Belko", "phone": "+2348012345678"}
)
print(f"Duplicate Probability: {pair_result.probability:.3f}")
print(f"Routing Decision:      {pair_result.tier.value}")
print(f"Prediction Set:        {pair_result.prediction_set}")
```

---

## 3. CLI Usage

```bash
# Deduplicate a CSV file and output JSON audit trail
wdedup deduplicate --input farmers.csv --output dedup_results.json

# Display the auditable MARS equation for a model
wdedup show-equation --model text_only_v1
wdedup show-equation --model text_phone_v1
wdedup show-equation --model text_phone_geo_v1
```
